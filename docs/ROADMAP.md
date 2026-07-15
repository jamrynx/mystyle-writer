# MyStyle Writer — 7-Day Roadmap (One Commit Per Day)

Rule: even if work finishes early, each day's scope is committed on its own day.
This produces a clean, honest-looking commit history that tells the story of the build.

---

## Day 1 (Sat) — Project scaffold
**Scope:** Repo structure, README, requirements.txt, .gitignore, roadmap, config skeleton.
**Commit:** `chore: project scaffold, docs, and dependency spec`
**Definition of done:** Repo clones and `pip install -r requirements.txt` succeeds.

## Day 2 (Sun) — Dataset collection + preparation pipeline
**Scope:** Collect raw style corpus into `data/raw/`. Write `src/prepare_data.py` —
cleaning, chunking (~512-token passages), and a 80/10/10 train/val/held-out split
written as JSONL to `data/processed/`. Held-out split is sacred: never trains, only evaluates.
**Commit:** `feat(data): raw corpus + cleaning/chunking/split pipeline`
**Definition of done:** `python src/prepare_data.py` produces train.jsonl / val.jsonl / heldout.jsonl with printed stats (chunk count, avg tokens).

## Day 3 (Mon) — Training notebook, first smoke-test run
**Scope:** `notebooks/train_lora.ipynb` — load base model in 4-bit, attach LoRA
(r=16, alpha=32, target attention projections), train 1 epoch on a small slice
to prove the pipeline end to end. Save adapter to `models/adapters/smoke-test/`.
**Commit:** `feat(train): LoRA training notebook + smoke-test adapter run`
**Definition of done:** Loss curve decreases; adapter saves and reloads for inference.

## Day 4 (Tue) — Full training run + inference module
**Scope:** Full fine-tune on the complete train split (2–3 epochs, early stop on val loss).
Write `src/inference.py`: one function that generates from base, one from base+adapter,
same sampling params for a fair comparison.
**Commit:** `feat(train): full fine-tune run + unified inference module`
**Definition of done:** Same prompt through both paths shows a visible style difference.

## Day 5 (Wed) — Streamlit comparison app
**Scope:** `app/streamlit_app.py` — prompt box, side-by-side Base vs Fine-Tuned panels,
sampling controls (temperature, max tokens), and a "sample prompts" dropdown.
**Commit:** `feat(app): side-by-side base vs fine-tuned Streamlit UI`
**Definition of done:** App runs locally; a non-technical viewer can see the style shift.

## Day 6 (Thu) — Quantitative style evaluation (Outcome #3)
**Scope:** `evaluation/style_eval.py` — score generated text against the held-out set:
(a) embedding centroid similarity to held-out style samples,
(b) lexical style features (avg sentence length, type-token ratio, punctuation profile).
Report base vs fine-tuned scores in a table; add results to README.
**Commit:** `feat(eval): style-match scoring vs held-out samples + results`
**Definition of done:** A single number (or small table) proves the model moved toward the target style.

## Day 7 (Fri) — Polish, deploy, document
**Scope:** Deploy the Streamlit app (Streamlit Community Cloud or Hugging Face Spaces),
final README with results screenshots, limitations section, and a short demo GIF.
**Commit:** `docs: results, deployment link, demo, and final report polish`
**Definition of done:** Public link works; README alone tells the full story for faculty review.

---

## Stretch goals (only after Day 7)
- Zimbabwean-corpus variant as a second adapter, switchable in the app
- LoRA rank ablation (r=8 vs 16 vs 32) with eval scores
- Push adapter to Hugging Face Hub with a model card
