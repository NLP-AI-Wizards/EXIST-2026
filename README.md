# EXIST-2026

## Install dependencies

Stop using `pip` and start using `uv`:

```bash
uv sync
```

# Multimodal Meme Sexism Detection Architecture

```
                         ┌─────────────────────┐
                         │      MEME INPUT     │
                         │  Image + Text +     │
                         │ Annotators + Sensors│
                         └──────────┬──────────┘
                                    │
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        │                           │                           │

 ┌──────────────┐          ┌─────────────────┐           ┌───────────────────┐
 │   IMAGE      │          │      TEXT        │           │    ANNOTATORS     │
 │   (JPG)      │          │  Meme caption    │           │ Annotator IDs     │
 └──────┬───────┘          └────────┬─────────┘           └────────┬──────────┘
        │                            │                              │
        │                            │                              │
   ┌────▼─────┐               ┌──────▼──────┐                ┌──────▼──────┐
   │ CNN /    │               │ BERT / LLM  │                │ Embedding   │
   │ ViT      │               │ Encoder     │                │ Layer       │
   └────┬─────┘               └──────┬──────┘                └──────┬──────┘
        │                             │                              │
        │                             │                              │
        ▼                             ▼                              ▼
   Image Emb                    Text Token Emb                 Annotator Emb



                         ┌───────────────────────────┐
                         │       SENSOR MODULE        │
                         └────────────┬──────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               │                      │                      │

        ┌─────────────┐       ┌──────────────┐       ┌─────────────┐
        │ EEG (16×5)  │       │ Eye Tracking │       │ Heart Rate  │
        │ bands/chans │       │ 24 features  │       │ 4 features  │
        └──────┬──────┘       └──────┬───────┘       └──────┬──────┘
               │                     │                      │
         ┌─────▼─────┐         ┌─────▼─────┐          ┌─────▼─────┐
         │ CNN       │         │ MLP       │          │ MLP       │
         └─────┬─────┘         └─────┬─────┘          └─────┬─────┘
               │                     │                      │
               └─────────────┬───────┴──────────────┬───────┘
                             ▼                      ▼
                        Physiological Embedding (concat)



═══════════════════════════════════════════════════════════
                MULTIMODAL FUSION
═══════════════════════════════════════════════════════════


Option A — Feature Fusion (baseline)

[image_emb | text_emb | annotator_emb | physio_emb]
                        │
                        ▼
                     MLP Head
                        │
                        ▼
             Sexism probability / distribution



Option B — Multimodal Transformer (cross-attention)

image tokens
text tokens
annotator token
physio token
        │
        ▼
  Multimodal Transformer
 (cross-modal attention)
        │
        ▼
     Classification Head



Option C — Modalities as Tokens

[IMG_token, EEG_token, ET_token, HR_token, ANNOT_token, text_tokens]
                        │
                        ▼
            Transformer Encoder
           (joint self-attention)
                        │
                        ▼
                  Classifier
```

---

# Output (LeWiDi)

Instead of predicting a single label:

```
YES / NO
```

the model predicts **annotator label distribution**

Example:

```
[YES YES YES NO NO YES]
→ target = 5/6 = 0.83
```

Possible training objectives:

* distribution classification (7 classes)
* regression on probability
* KL divergence to vote distribution

---

# Training

End-to-end training updates:

* image encoder
* text encoder
* sensor encoders
* annotator embeddings
* multimodal fusion layers

We model **interactions between semantics, perception, annotator bias, and physiological response**.