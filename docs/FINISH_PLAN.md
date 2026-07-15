# FINISH PLAN — Today + Tomorrow (deadline compression)

Original 7-day roadmap compressed to 2 days. Multiple commits per day is fine —
what matters is each commit is a coherent unit of work.

---

## TODAY — 4 commits (all code is already written; only Colab runs are on you)

### Commit 1 — data (if not already pushed)
```
git add data/ src/prepare_data.py .gitignore
git commit -m "feat(data): Wilde corpus + cleaning/chunking/split pipeline with boilerplate filter"
```

### Commit 2 — training notebook + inference module
```
git add notebooks/train_lora.ipynb src/inference.py
git commit -m "feat(train): Colab LoRA notebook + unified base/adapter inference module"
```

### Commit 3 — evaluation (Outcome #3)
```
git add evaluation/style_eval.py
git commit -m "feat(eval): style-match scoring vs held-out split (embedding centroid + lexical profile)"
```

### Commit 4 — Streamlit app
```
git add app/streamlit_app.py
git commit -m "feat(app): side-by-side base vs fine-tuned UI with live + demo modes"
git push
```

### TONIGHT — the one thing only you can do: RUN COLAB
1. Open notebooks/train_lora.ipynb in Colab, T4 runtime
2. Edit REPO_URL in cell 1, set SMOKE_TEST = True, Run All (~15 min)
3. If loss decreases: set SMOKE_TEST = False, Run All again (full train, ~30-60 min)
4. Keep running: Section 7 cells (batch generation + evaluation, ~15 min)
5. Download: wilde-full-adapter.zip and eval-outputs.zip

---

## TOMORROW (deadline day) — 3 commits

### Commit 5 — training + eval results
Unzip eval-outputs.zip into the repo (evaluation/*.jsonl, evaluation/results.json).
Save the executed notebook (File -> Download .ipynb) over notebooks/train_lora.ipynb.
```
git add notebooks/train_lora.ipynb evaluation/generated_*.jsonl evaluation/results.json
git commit -m "feat(train): full Wilde fine-tune — executed notebook, generations, and eval results"
```

### Commit 6 — deploy demo mode
Demo mode needs no GPU: push repo to Hugging Face Spaces (Streamlit SDK) or
Streamlit Community Cloud. The app auto-detects generated_*.jsonl and serves
pre-generated comparisons + the eval metric in the sidebar.
```
git add <any deploy config>
git commit -m "chore(deploy): demo-mode deployment config"
```

### Commit 7 — final README
Add: results table from evaluation/results.json, one before/after example,
deployment link, limitations paragraph (small model, single-author corpus,
style != content safety).
```
git add README.md
git commit -m "docs: results, live demo link, limitations — final report"
git push
```

---

## If something fails tonight
- Colab OOM -> switch BASE_MODEL in src/config.py to TinyLlama-1.1B and rerun
- Colab disconnects mid-train -> reduce EPOCHS to 2; smoke-test adapter is
  still a valid (weaker) demo
- No time for deploy -> demo mode running locally + screenshots in README
  still satisfies all three project outcomes
