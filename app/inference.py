"""
Loading and inference helpers for the Transformer NMT model.

Expected files in `model_dir`:
  - final_nmt.pt   : torch.save dict with keys
        'model_state', 'config' {src_vocab, tgt_vocab, d_model, n_layers,
        n_heads, d_ff, dropout}, and optional 'history', 'best_epoch',
        'best_val', 'bleu_greedy', 'bleu_beam5'.
  - spm_en.model   : SentencePiece BPE model for English.
  - spm_ar.model   : SentencePiece BPE model for Arabic.

`best_nmt.pt` is also accepted as a fallback for `model_state`, but only
together with a `final_nmt.pt` (or its config), since `best_nmt.pt`
saves only the raw `state_dict` without architecture metadata.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import torch

from .model import EOS_IDX, PAD_IDX, SOS_IDX, Transformer


# ---------------------------------------------------------------------------
# Preprocessing — must match notebook Cell 5 exactly
# ---------------------------------------------------------------------------
_EN_PUNCT = re.compile(r"([.!?,;:])")
_EN_KEEP = re.compile(r"[^a-z0-9?.!,;:' ]")
_WS = re.compile(r"\s+")
_AR_PUNCT = re.compile(r"([?!.،,؛])")


def preprocess_en(text: str) -> str:
    text = text.lower().strip()
    text = _EN_PUNCT.sub(r" \1 ", text)
    text = _EN_KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def preprocess_ar(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    text = _AR_PUNCT.sub(r" \1 ", text)
    return _WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Loaded-model bundle
# ---------------------------------------------------------------------------
@dataclass
class TranslationModel:
    model: Transformer
    sp_en: "object"  # sentencepiece.SentencePieceProcessor
    sp_ar: "object"
    config: dict
    device: torch.device
    metadata: dict  # best_epoch, best_val, bleu_greedy, bleu_beam5, num_params

    def translate(
        self,
        sentence: str,
        beam_width: int = 5,
        max_len: int = 100,
        alpha: float = 0.6,
    ) -> str:
        sentence = preprocess_en(sentence)
        if not sentence:
            return ""
        ids = self.sp_en.encode(sentence) + [EOS_IDX]
        src = torch.tensor(ids, dtype=torch.long)
        if beam_width <= 1:
            out_ids = self.model.translate_greedy(src, self.device, max_len=max_len)
        else:
            out_ids = self.model.translate_beam(
                src, self.device, beam_width=beam_width, max_len=max_len, alpha=alpha
            )
        out_ids = [i for i in out_ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        return self.sp_ar.decode(out_ids)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    pass


def _resolve_paths(model_dir: str) -> dict:
    return {
        "final": os.path.join(model_dir, "final_nmt.pt"),
        "best": os.path.join(model_dir, "best_nmt.pt"),
        "spm_en": os.path.join(model_dir, "spm_en.model"),
        "spm_ar": os.path.join(model_dir, "spm_ar.model"),
    }


def check_model_files(model_dir: str) -> dict:
    """Return a status dict describing which files are present/missing."""
    paths = _resolve_paths(model_dir)
    present = {k: os.path.isfile(v) for k, v in paths.items()}
    has_ckpt = present["final"] or present["best"]
    ready = has_ckpt and present["spm_en"] and present["spm_ar"]
    missing = [k for k, ok in present.items() if not ok]
    return {
        "model_dir": os.path.abspath(model_dir),
        "paths": paths,
        "present": present,
        "ready": ready,
        "missing": missing,
    }


def load_model(model_dir: str, device: Optional[torch.device] = None) -> TranslationModel:
    """Build the Transformer, load weights, load both SentencePiece tokenizers."""
    import sentencepiece as spm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    status = check_model_files(model_dir)
    paths = status["paths"]

    if not (status["present"]["final"] or status["present"]["best"]):
        raise ModelLoadError(
            f"No checkpoint found in {status['model_dir']}. "
            f"Expected `final_nmt.pt` (preferred) or `best_nmt.pt`."
        )
    if not status["present"]["spm_en"] or not status["present"]["spm_ar"]:
        raise ModelLoadError(
            f"SentencePiece models missing in {status['model_dir']}. "
            f"Need both `spm_en.model` and `spm_ar.model`."
        )

    if status["present"]["final"]:
        ckpt = torch.load(paths["final"], map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "config" not in ckpt or "model_state" not in ckpt:
            raise ModelLoadError(
                "`final_nmt.pt` is not in the expected format "
                "(missing 'config' or 'model_state')."
            )
        config = dict(ckpt["config"])
        state = ckpt["model_state"]
        meta = {
            "best_epoch": ckpt.get("best_epoch"),
            "best_val": ckpt.get("best_val"),
            "bleu_greedy": ckpt.get("bleu_greedy"),
            "bleu_beam5": ckpt.get("bleu_beam5"),
            "history": ckpt.get("history"),
            "checkpoint_file": "final_nmt.pt",
        }
    else:
        raise ModelLoadError(
            "Only `best_nmt.pt` was found; it stores raw weights without an "
            "architecture config. Save and copy `final_nmt.pt` from the "
            "notebook's last cell so the GUI knows the model dimensions."
        )

    model = Transformer(**config).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys or unexpected_keys:
        raise ModelLoadError(
            "state_dict does not match architecture.\n"
            f"  missing keys   : {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}\n"
            f"  unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
        )
    model.eval()

    sp_en = spm.SentencePieceProcessor(model_file=paths["spm_en"])
    sp_ar = spm.SentencePieceProcessor(model_file=paths["spm_ar"])

    meta["num_params"] = sum(p.numel() for p in model.parameters())
    meta["en_vocab"] = sp_en.get_piece_size()
    meta["ar_vocab"] = sp_ar.get_piece_size()

    return TranslationModel(
        model=model,
        sp_en=sp_en,
        sp_ar=sp_ar,
        config=config,
        device=device,
        metadata=meta,
    )
