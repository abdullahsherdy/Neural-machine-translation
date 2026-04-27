"""
Gradio web GUI for the English -> Arabic Transformer NMT model.

Run from the project root:

    python -m app.gui

By default the app looks for the trained model files in `./output/`.
Override with the env var NMT_MODEL_DIR or the `--model-dir` CLI flag.

Expected files in the model directory:
    final_nmt.pt    spm_en.model    spm_ar.model

If the files are not yet there, the GUI still launches with a clear
"model not loaded" banner. Drop the files in, click "Reload model",
and start translating.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from typing import Optional

import gradio as gr
import torch

from .inference import (
    ModelLoadError,
    TranslationModel,
    check_model_files,
    load_model,
)


# ---------------------------------------------------------------------------
# Mutable global model handle. Re-assigned by `do_reload`.
# ---------------------------------------------------------------------------
_MODEL: Optional[TranslationModel] = None
_LOAD_ERROR: Optional[str] = None
_MODEL_DIR: str = ""


# ---------------------------------------------------------------------------
# Status / info rendering
# ---------------------------------------------------------------------------
def _status_md() -> str:
    status = check_model_files(_MODEL_DIR)
    lines = [f"**Model directory:** `{status['model_dir']}`"]
    if _MODEL is not None:
        lines.append(
            f"**Status:** Model loaded on `{_MODEL.device}` "
            f"({_MODEL.metadata['num_params']:,} params)."
        )
    else:
        if _LOAD_ERROR:
            lines.append(f"**Status:** Not loaded - {_LOAD_ERROR}")
        elif not status["ready"]:
            missing = ", ".join(f"`{m}`" for m in status["missing"])
            lines.append(
                f"**Status:** Waiting for files. Missing: {missing}. "
                f"Drop them into the model directory and click **Reload model**."
            )
        else:
            lines.append("**Status:** Files found but not loaded yet.")
    return "\n\n".join(lines)


def _info_md() -> str:
    if _MODEL is None:
        status = check_model_files(_MODEL_DIR)
        rows = ["| File | Present |", "|---|---|"]
        for name, ok in status["present"].items():
            rows.append(f"| `{os.path.basename(status['paths'][name])}` | {'yes' if ok else 'no'} |")
        return (
            "### Model not loaded\n\n"
            f"Looking for files in `{status['model_dir']}`.\n\n"
            + "\n".join(rows)
            + "\n\nOnce the Kaggle run finishes, copy `final_nmt.pt`, "
            "`spm_en.model` and `spm_ar.model` here and click **Reload model**."
        )
    cfg = _MODEL.config
    md = _MODEL.metadata

    def _fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows = [
        ("Device", str(_MODEL.device)),
        ("Total parameters", f"{md['num_params']:,}"),
        ("Encoder layers", cfg["n_layers"]),
        ("Decoder layers", cfg["n_layers"]),
        ("Model dim (d_model)", cfg["d_model"]),
        ("Attention heads", cfg["n_heads"]),
        ("Feed-forward dim", cfg["d_ff"]),
        ("Dropout", cfg["dropout"]),
        ("EN vocab (config)", cfg["src_vocab"]),
        ("AR vocab (config)", cfg["tgt_vocab"]),
        ("EN vocab (loaded SP)", md.get("en_vocab")),
        ("AR vocab (loaded SP)", md.get("ar_vocab")),
        ("Best epoch", md.get("best_epoch")),
        ("Best val loss", md.get("best_val")),
        ("BLEU-4 (greedy)", md.get("bleu_greedy")),
        ("BLEU-4 (beam-5)", md.get("bleu_beam5")),
        ("Checkpoint file", md.get("checkpoint_file")),
    ]
    body = ["| Field | Value |", "|---|---|"]
    body += [f"| {k} | {_fmt(v)} |" for k, v in rows]
    return "### Loaded model\n\n" + "\n".join(body)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def do_reload():
    global _MODEL, _LOAD_ERROR
    try:
        _MODEL = load_model(_MODEL_DIR)
        _LOAD_ERROR = None
    except ModelLoadError as e:
        _MODEL = None
        _LOAD_ERROR = str(e)
    except Exception as e:
        _MODEL = None
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
    return _status_md(), _info_md()


def do_translate(sentence: str, beam_width: int, max_len: int, history):
    history = history or []
    if not sentence or not sentence.strip():
        return "", history, _history_to_rows(history)
    if _MODEL is None:
        msg = "Model is not loaded. Open the **Model Info** tab and click **Reload model**."
        return msg, history, _history_to_rows(history)
    try:
        out = _MODEL.translate(
            sentence,
            beam_width=int(beam_width),
            max_len=int(max_len),
        )
    except Exception as e:
        return f"Translation failed: {type(e).__name__}: {e}", history, _history_to_rows(history)

    decode = "greedy" if int(beam_width) <= 1 else f"beam-{int(beam_width)}"
    history = history + [
        {
            "ts": _dt.datetime.now().strftime("%H:%M:%S"),
            "en": sentence.strip(),
            "ar": out,
            "decode": decode,
        }
    ]
    return out, history, _history_to_rows(history)


def do_clear_history():
    return [], _history_to_rows([])


def _history_to_rows(history):
    return [[h["ts"], h["en"], h["ar"], h["decode"]] for h in history]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
EXAMPLE_SENTENCES = [
    "i love you .",
    "the book is on the table .",
    "hello , how are you today ?",
    "the government announced new economic reforms .",
    "she does not know him .",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EN -> AR Transformer NMT", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# English -> Arabic Neural Machine Translation\n"
            "Transformer encoder-decoder, trained from scratch on OPUS-100 (en-ar)."
        )
        status_box = gr.Markdown(_status_md())

        with gr.Tabs():
            # ------------------------------ Translate
            with gr.Tab("Translate"):
                with gr.Row():
                    with gr.Column():
                        en_in = gr.Textbox(
                            label="English",
                            placeholder="Type an English sentence...",
                            lines=4,
                        )
                        with gr.Accordion("Decoding options", open=False):
                            beam = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Beam width (1 = greedy)",
                            )
                            max_len = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=100,
                                step=10,
                                label="Max output length",
                            )
                        translate_btn = gr.Button("Translate", variant="primary")
                    with gr.Column():
                        ar_out = gr.Textbox(
                            label="Arabic",
                            lines=4,
                            rtl=True,
                            show_copy_button=True,
                        )

                gr.Examples(
                    examples=[[s] for s in EXAMPLE_SENTENCES],
                    inputs=en_in,
                    label="Example sentences",
                )

            # ------------------------------ History
            with gr.Tab("History"):
                history_state = gr.State([])
                history_table = gr.Dataframe(
                    headers=["Time", "English", "Arabic", "Decoding"],
                    datatype=["str", "str", "str", "str"],
                    row_count=(0, "dynamic"),
                    col_count=(4, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="Translation history",
                )
                clear_btn = gr.Button("Clear history")

            # ------------------------------ Model info
            with gr.Tab("Model info"):
                info_md = gr.Markdown(_info_md())
                reload_btn = gr.Button("Reload model")

        translate_btn.click(
            do_translate,
            inputs=[en_in, beam, max_len, history_state],
            outputs=[ar_out, history_state, history_table],
        )
        en_in.submit(
            do_translate,
            inputs=[en_in, beam, max_len, history_state],
            outputs=[ar_out, history_state, history_table],
        )
        clear_btn.click(do_clear_history, outputs=[history_state, history_table])
        reload_btn.click(do_reload, outputs=[status_box, info_md])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EN->AR Transformer NMT GUI")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("NMT_MODEL_DIR", "output"),
        help="Directory containing final_nmt.pt, spm_en.model, spm_ar.model "
        "(default: ./output)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (useful in Colab/Kaggle).",
    )
    args = parser.parse_args()

    global _MODEL_DIR
    _MODEL_DIR = args.model_dir
    os.makedirs(_MODEL_DIR, exist_ok=True)

    # Try to load on startup; failures are non-fatal so the GUI still opens.
    try:
        do_reload()
    except Exception:
        pass

    print(f"Model dir : {os.path.abspath(_MODEL_DIR)}")
    print(f"Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if _MODEL is None:
        print(f"Model     : not loaded ({_LOAD_ERROR})")
    else:
        print(f"Model     : loaded ({_MODEL.metadata['num_params']:,} params)")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
