# Neural Machine Translation — English to Arabic

Two implementations of sequence-to-sequence neural machine translation trained on an English-Arabic parallel corpus. The first uses a recurrent encoder-decoder with Bahdanau attention, the second replaces the recurrent components entirely with a Transformer.

Both notebooks are self-contained and run on Google Colab with a free T4 GPU.

---

## Dataset

The dataset is a tab-separated file of English-Arabic sentence pairs (`ara_.txt`), containing roughly 10,700 pairs ranging from single-word phrases to complex multi-clause sentences. It is derived from the Tatoeba corpus.

Format:
```
Hello!    مرحباً.
Run!      اركض!
I know.   أعرف.
```

---

## Notebooks

### 1. RNN with Bahdanau Attention

**File:** `nmt_english_arabic_colab.ipynb`

A classic sequence-to-sequence model where both the encoder and decoder are recurrent networks. The encoder reads the source sentence with a bidirectional GRU and produces a sequence of hidden states. At each decoder step, the attention mechanism computes a weighted sum over those states, allowing the model to focus on different parts of the source dynamically.

Architecture:

```
Source tokens
    -> Embedding
    -> Bidirectional GRU Encoder  -> h_1 ... h_n
    -> Bahdanau Attention          -> score(s_t, h_j) = v^T tanh(W1 s_t + W2 h_j)
    -> context_t = sum(alpha_j * h_j)
    -> GRU Decoder  [embed(y); context_t]  -> s_t  -> FC  -> prediction
```

Key components:

- Bidirectional GRU encoder with forward/backward state concatenation
- Additive (Bahdanau) attention scoring
- Teacher forcing during training with configurable ratio
- Greedy and beam search decoding

Reference: Bahdanau et al., 2015. *Neural Machine Translation by Jointly Learning to Align and Translate.*

---

### 2. Transformer Encoder-Decoder

**File:** `transformer_nmt_colab.ipynb`

A full implementation of the original Transformer architecture from scratch. All recurrence is eliminated. Instead, every token in the sequence attends directly to every other token in parallel through multi-head self-attention. The decoder additionally attends to the encoder output through cross-attention at each layer.

Architecture:

```
Source tokens
    -> Embedding + Positional Encoding
    -> Encoder (N layers)
          Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm
    -> memory

Target tokens (shifted right)
    -> Embedding + Positional Encoding
    -> Decoder (N layers)
          Masked Self-Attention -> Add & Norm
          Cross-Attention over memory -> Add & Norm
          Feed-Forward -> Add & Norm
    -> Linear -> Softmax -> prediction
```

Key components:

- Sinusoidal positional encoding
- Scaled dot-product multi-head attention
- Separate PAD mask and causal (triangular) mask for the decoder
- Noam learning rate schedule with linear warmup
- Label smoothing loss
- Beam search with length penalty
- Per-head cross-attention visualisation

Reference: Vaswani et al., 2017. *Attention Is All You Need.*

---

## Getting Started

**Requirements:** A Google account with access to Colab. No local setup needed.

1. Open the notebook in Google Colab.
2. Go to `Runtime -> Change runtime type` and select `T4 GPU`.
3. Run cell 1. A file upload dialog will appear — upload `ara_.txt`.
4. Run all remaining cells in order.

The notebooks detect available hardware automatically and adjust model size and batch size accordingly.

---

## Model Configurations

Both notebooks switch between a smaller CPU config and a larger GPU config at runtime.

| Parameter | RNN (GPU) | Transformer (GPU) |
|---|---|---|
| Max sentence length | 20 tokens | 50 tokens |
| Embedding dimension | 256 | 256 |
| Hidden / model dim | 512 | 256 |
| Encoder layers | 1 (BiGRU) | 3 |
| Decoder layers | 1 (GRU) | 3 |
| Attention heads | 1 | 8 |
| Feed-forward dim | — | 512 |
| Batch size | 64 | 128 |
| Epochs | 30 | 30 |
| Optimizer | Adam | Adam (Noam schedule) |
| Loss | Cross-entropy | Label-smoothed CE |

---

## Outputs

Both notebooks save the following to `/content/` at the end of training:

| File | Description |
|---|---|
| `best_*.pt` | Model checkpoint (weights, vocabularies, config) |
| `training_curves.png` | Loss and perplexity over epochs |
| `attention_heatmaps.png` | Cross-attention for sample sentences |
| `training_history.json` | Numeric training history |

The Transformer notebook also saves:

| File | Description |
|---|---|
| `per_head_attention.png` | Separate heatmap for each attention head |
| `lr_schedule.png` | Noam learning rate curve |

All files are downloaded to your local machine via `google.colab.files.download` at the end of the final cell.

---

## Evaluation

Both notebooks compute BLEU-4 using an implementation with add-1 smoothing, evaluated on 300-400 held-out validation pairs. Results are broken down by source sentence length.

---

## Comparison

| | RNN + Bahdanau | Transformer |
|---|---|---|
| Alignment | Single-head additive | Multi-head dot-product |
| Context | Sequential GRU states | All positions in parallel |
| Long-range dependencies | Degrades with sequence length | Direct via attention |
| Training speed on GPU | Slower (sequential steps) | Faster (fully parallelisable) |
| BLEU (same data, same epochs) | Lower baseline | Higher by several points |
| Interpretability | 1 attention matrix per step | Per-layer, per-head matrices |

---

## Limitations

- The dataset is small (~10,700 pairs). Translation quality improves substantially with more data such as the OPUS EN-AR corpus.
- Tokenisation is whitespace-based. Arabic morphology is complex and subword tokenisation (SentencePiece or BPE) would improve coverage of rare word forms.
- Both models are trained from scratch. For production use, fine-tuning a pre-trained model such as Helsinki-NLP/opus-mt-en-ar would give better results with less compute.

---

## Repository Structure

```
.
├── nmt_english_arabic_colab.ipynb     # RNN + Bahdanau attention
├── transformer_nmt_colab.ipynb        # Transformer encoder-decoder
├── ara_.txt                           # Parallel corpus (add manually)
└── README.md
```

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015.*
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS 2017.*
- Tatoeba Project. https://tatoeba.org
