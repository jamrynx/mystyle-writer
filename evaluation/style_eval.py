"""Quantitative style-match evaluation (Project Outcome #3).

Scores base vs fine-tuned generations against the HELD-OUT style samples
(never seen in training) on two independent axes:

1. Embedding style score — cosine similarity between each generation and the
   centroid of held-out Wilde chunk embeddings (all-MiniLM-L6-v2).
   Higher = closer to the target style distribution.

2. Lexical style profile — avg sentence length, type-token ratio, punctuation
   rates (em-dash/semicolon/exclamation), dialogue density. Reported for
   held-out reference vs base vs tuned so the shift is inspectable.

Runs on CPU. Inputs come from `python src/inference.py --batch` (see README).

Usage:
    python evaluation/style_eval.py \
        --base evaluation/generated_base.jsonl \
        --tuned evaluation/generated_tuned.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import config

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------ io
def read_jsonl_texts(path: str | Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


# ------------------------------------------------------------ axis 1: embeddings
def embedding_style_scores(heldout: list[str], conditions: dict[str, list[str]]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    emb_ref = model.encode(heldout, normalize_embeddings=True, show_progress_bar=False)
    centroid = emb_ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    def score(texts):
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ centroid
        return float(sims.mean()), float(sims.std())

    return {name: score(texts) for name, texts in conditions.items()}


# ------------------------------------------------------------ axis 2: lexical profile
def lexical_profile(texts: list[str]) -> dict:
    joined = " ".join(texts)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
    words = joined.split()
    n_words = max(1, len(words))
    n_sents = max(1, len(sentences))
    return {
        "avg_sentence_len": round(n_words / n_sents, 1),
        "type_token_ratio": round(len({w.lower().strip('.,;!?"\u2014') for w in words}) / n_words, 3),
        "semicolons_per_1k": round(joined.count(";") / n_words * 1000, 2),
        "exclaims_per_1k": round(joined.count("!") / n_words * 1000, 2),
        "dialogue_marks_per_1k": round((joined.count('"') + joined.count("\u201c")) / n_words * 1000, 2),
    }


# ------------------------------------------------------------ report
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="evaluation/generated_base.jsonl")
    ap.add_argument("--tuned", default="evaluation/generated_tuned.jsonl")
    ap.add_argument("--prompted", default="evaluation/generated_prompted.jsonl",
                    help="style-prompted base generations (optional; pass '' to skip)")
    ap.add_argument("--out", default="evaluation/results.json")
    args = ap.parse_args()

    heldout = read_jsonl_texts(config.HELDOUT_FILE)
    conditions = {"base": read_jsonl_texts(args.base)}
    if args.prompted and Path(args.prompted).exists():
        conditions["prompted"] = read_jsonl_texts(args.prompted)
    conditions["tuned"] = read_jsonl_texts(args.tuned)
    print("held-out:", len(heldout), "|",
          " | ".join(f"{k}: {len(v)}" for k, v in conditions.items()), "\n")

    emb = embedding_style_scores(heldout, conditions)
    delta = emb["tuned"][0] - emb["base"][0]

    print("=== Embedding style score (cosine to held-out centroid) ===")
    for name in conditions:
        mu, sd = emb[name]
        print(f"  {name:<10} {mu:.4f} ± {sd:.4f}")
    print(f"  tuned - base delta: {delta:+.4f}  "
          f"{'(moved TOWARD target style)' if delta > 0 else '(no improvement)'}")
    if "prompted" in emb:
        edge = emb["tuned"][0] - emb["prompted"][0]
        print(f"  tuned - prompted:   {edge:+.4f}  "
              "(fine-tuning edge over just prompting — Unit 3 decision framework)")
    print()

    profiles = {"heldout_reference": lexical_profile(heldout)}
    profiles.update({name: lexical_profile(texts) for name, texts in conditions.items()})

    print("=== Lexical style profile ===")
    keys = list(profiles["heldout_reference"].keys())
    cols = list(profiles.keys())
    header = f"{'metric':<24}" + "".join(f"{c[:10]:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{profiles[c][k]:>12}" for c in cols))

    results = {
        "embedding_score": {name: {"mean": emb[name][0], "std": emb[name][1]} for name in emb},
        "delta_tuned_vs_base": delta,
        "lexical_profiles": profiles,
        "counts": {"heldout": len(heldout), **{k: len(v) for k, v in conditions.items()}},
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
