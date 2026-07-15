# MyStyle Writer — Style Fine-Tuning with LoRA

Fine-tune a small open-source language model with **QLoRA** (4-bit quantized base + LoRA adapters, via PEFT) so it imitates a chosen writing style, then compare **base vs fine-tuned** outputs side by side in a Streamlit app, with a quantitative style-match evaluation on held-out samples.

> Summer Training Project — AI Engineer Launchpad: Mastering LLMs and Agentic AI (Web ID 1378)
> Project #27 · Source: Research-paper Based

## Why this project

Most LLM apps live at the API layer — prompt in, text out. This project goes under the hood: dataset curation, tokenization, parameter-efficient fine-tuning (QLoRA), adapter serving, and quantitative evaluation. It runs entirely on a **free Colab T4 GPU** (or a local 6GB card).

## The fine-tune vs. prompt decision framework (Unit 3)

Could we get Wilde's style by just prompting? This project answers that **empirically** instead of assuming. The evaluation compares three conditions against held-out Wilde samples:

1. **Base model** — no style guidance (floor)
2. **Style-prompted base** — "Write in the style of Oscar Wilde..." (the cheap alternative)
3. **QLoRA fine-tuned** — behavior reshaped by training

If (3) beats (2) on the held-out style score, fine-tuning earned its cost for this task. If not, prompting would have been the honest recommendation. Style transfer is a good candidate for fine-tuning precisely because style is diffuse — it lives in rhythm, vocabulary distribution, and sentence structure that a one-line instruction can only approximate. RAG is the wrong tool here entirely: this is a *behavior* problem, not a *knowledge* problem.

## Course outcome mapping

| Course outcome | Where this project demonstrates it |
|---|---|
| CO3 — PEFT fine-tuning (LoRA/QLoRA), quantization, dataset preparation | Training notebook (4-bit NF4 + LoRA r=16), `prepare_data.py` cleaning/chunking/split pipeline |
| CO5 — evaluation of LLM systems | Three-way held-out evaluation: embedding style score + lexical profile |
| CO6 — deployment | Streamlit app (live + demo modes), deployable to HF Spaces / Streamlit Cloud |

## Project Outcomes

1. Complete a LoRA fine-tune of a small model and serve before-and-after comparisons.
2. Show a clear, measurable style shift after fine-tuning.
3. Add an evaluation that quantifies style match against held-out samples.

## Architecture

```
data/raw ──► src/prepare_data.py ──► data/processed (train/val/held-out splits)
                                          │
                                          ▼
              notebooks/train_lora.ipynb (Colab, T4)
                                          │
                                          ▼
                            models/adapters/<run-name>/
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
              app/streamlit_app.py                evaluation/style_eval.py
           (base vs tuned, side by side)      (style-match score on held-out set)
```

## Repository Structure

```
mystyle-writer/
├── data/
│   ├── raw/           # source texts (never edited by hand after collection)
│   └── processed/     # cleaned, chunked, split JSONL (train / val / heldout)
├── notebooks/
│   └── train_lora.ipynb    # Colab training notebook (T4-friendly)
├── src/
│   ├── prepare_data.py     # clean, chunk, split raw text into JSONL
│   ├── config.py           # single source of truth for model & training params
│   └── inference.py        # load base model +/- adapter, generate text
├── app/
│   └── streamlit_app.py    # side-by-side base vs fine-tuned comparison UI
├── evaluation/
│   └── style_eval.py       # quantitative style-match scoring vs held-out set
├── models/
│   └── adapters/           # saved LoRA adapters (gitignored except .gitkeep)
├── docs/
│   └── ROADMAP.md          # 7-day plan, one commit per day
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) | Small enough for free T4, strong enough to show style shift |
| Fine-tuning | Hugging Face `transformers` + `peft` + `bitsandbytes` — **QLoRA** (4-bit NF4 base, LoRA r=16 adapters) | Parameter-efficient, fits 6GB VRAM |
| Data | JSONL chunks from a public-domain style corpus | Reproducible, license-safe |
| App | Streamlit | Fast to build, easy to deploy |
| Eval | Embedding-similarity style score + lexical style features | Quantifiable Outcome #3 |

## Quickstart

```bash
pip install -r requirements.txt
python src/prepare_data.py --input data/raw --out data/processed
# Training happens in notebooks/train_lora.ipynb on Colab (T4)
streamlit run app/streamlit_app.py
```

## Responsible AI notes

- All training data is public-domain text (Project Gutenberg mirrors); no scraped or licensed content.
- Style imitation of a long-deceased author's public-domain prose raises no impersonation concern; the same technique applied to a living writer would require their consent.
- The model inherits the base model's safety behavior; the adapter changes style, not safety guardrails. Known limitation: heavy style tuning can slightly degrade instruction-following — the evaluation's neutral prompts monitor for this.
- Period literature contains dated social attitudes; generations reflect style, not endorsed views.

## Status

See [docs/ROADMAP.md](docs/ROADMAP.md) for the day-by-day plan and [docs/FINISH_PLAN.md](docs/FINISH_PLAN.md) for the compressed 2-day schedule.
