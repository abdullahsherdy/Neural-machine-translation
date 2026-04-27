"""
Transformer architecture for English -> Arabic NMT.

This is a 1:1 mirror of the model classes defined in
`src/notebook32f2863234.ipynb` (Cell 8). The class names, attribute names
and forward-pass shapes are identical so a `state_dict` saved by the
notebook (`final_nmt.pt['model_state']` or `best_nmt.pt`) loads here
without any key remapping.

Key choices (must match the notebook exactly):
- Pre-Norm encoder/decoder layers
- GELU in the feed-forward block
- Sinusoidal positional encoding registered as a buffer
- Embeddings scaled by sqrt(d_model)
- Final LayerNorm after the encoder stack and after the decoder stack
- Output projection weight is tied to the decoder embedding
- bias=False on all attention projections
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        def split(x, w):
            return w(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q = split(query, self.w_q)
        K = split(key, self.w_k)
        V = split(value, self.w_v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(out), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        n1 = self.norm1(src)
        a, attn = self.self_attn(n1, n1, n1, src_mask)
        src = src + self.dropout(a)
        src = src + self.dropout(self.ffn(self.norm2(src)))
        return src, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        n1 = self.norm1(tgt)
        a, sa_w = self.self_attn(n1, n1, n1, tgt_mask)
        tgt = tgt + self.dropout(a)
        b, ca_w = self.cross_attn(self.norm2(tgt), memory, memory, src_mask)
        tgt = tgt + self.dropout(b)
        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt, sa_w, ca_w


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask):
        x = self.pe(self.embedding(src) * self.scale)
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attn_weights.append(attn)
        return self.norm(x), attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.pe(self.embedding(tgt) * self.scale)
        sa_all, ca_all = [], []
        for layer in self.layers:
            x, sa, ca = layer(x, memory, tgt_mask, src_mask)
            sa_all.append(sa)
            ca_all.append(ca)
        return self.norm(x), sa_all, ca_all


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab, bias=False)
        self.fc_out.weight = self.decoder.embedding.weight
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_src_mask(src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt):
        B, T = tgt.shape
        pad_m = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        return pad_m & causal.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory, enc_attn = self.encoder(src, src_mask)
        out, dec_self, dec_cross = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.fc_out(out), enc_attn, dec_cross

    @torch.no_grad()
    def translate_greedy(self, src_tensor: torch.Tensor, device: torch.device, max_len: int = 100):
        self.eval()
        src = src_tensor.unsqueeze(0).to(device)
        src_mask = self.make_src_mask(src)
        memory, _ = self.encoder(src, src_mask)
        tgt = torch.tensor([[SOS_IDX]], device=device)
        for _ in range(max_len):
            out, _, _ = self.decoder(tgt, memory, self.make_tgt_mask(tgt), src_mask)
            next_tok = self.fc_out(out[:, -1, :]).argmax(-1)
            if next_tok.item() == EOS_IDX:
                break
            tgt = torch.cat([tgt, next_tok.unsqueeze(0)], dim=1)
        return tgt[0, 1:].tolist()

    @torch.no_grad()
    def translate_beam(
        self,
        src_tensor: torch.Tensor,
        device: torch.device,
        beam_width: int = 5,
        max_len: int = 100,
        alpha: float = 0.6,
    ):
        self.eval()
        src = src_tensor.unsqueeze(0).to(device)
        src_mask = self.make_src_mask(src)
        memory, _ = self.encoder(src, src_mask)

        beams = [(0.0, [SOS_IDX])]
        done: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            candidates = []
            for log_p, toks in beams:
                if toks[-1] == EOS_IDX:
                    done.append((log_p, toks))
                    continue
                tgt_t = torch.tensor([toks], device=device)
                out, _, _ = self.decoder(
                    tgt_t, memory, self.make_tgt_mask(tgt_t), src_mask
                )
                lp = F.log_softmax(self.fc_out(out[:, -1, :]), dim=-1).squeeze(0)
                top_lp, top_idx = lp.topk(beam_width)
                for p, i in zip(top_lp.tolist(), top_idx.tolist()):
                    candidates.append((log_p + p, toks + [i]))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            if not beams or all(t[-1] == EOS_IDX for _, t in beams):
                done.extend(beams)
                break

        if not done:
            done = beams

        def length_penalized(item):
            lp, toks = item
            pen = ((5 + len(toks)) / 6) ** alpha
            return lp / pen

        _, best = max(done, key=length_penalized)
        return [t for t in best[1:] if t not in (EOS_IDX, SOS_IDX, PAD_IDX)]
