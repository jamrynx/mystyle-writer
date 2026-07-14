"""Clean, chunk, and split raw style-corpus text into JSONL datasets.

Usage:
    python src/prepare_data.py

Reads every .txt file in data/raw/, cleans it, chunks it into
~CHUNK_TOKENS passages with overlap, shuffles, and writes
train / val / heldout JSONL files to data/processed/.

The held-out split is EVAL-ONLY. It is never used in training,
so the style-match evaluation (Outcome #3) stays honest.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config


def clean_text(text: str) -> str:
    """Normalize whitespace and strip boilerplate-ish noise."""
    # Strip Project Gutenberg headers/footers if present
    start = re.search(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}", text, re.S)
    end = re.search(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}", text, re.S)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]

    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank-line runs
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse spaces
    return text.strip()


def approx_token_count(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per word) — good enough for chunk sizing."""
    return int(len(text.split()) * 1.3)


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Sliding-window chunking over words, sized by approximate tokens."""
    words = text.split()
    words_per_chunk = max(1, int(chunk_tokens / 1.3))
    step = max(1, words_per_chunk - int(overlap / 1.3))

    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + words_per_chunk])
        if approx_token_count(chunk) >= chunk_tokens * 0.5:  # drop tiny tail chunks
            chunks.append(chunk)
    return chunks


def main() -> None:
    raw_files = sorted(config.DATA_RAW.glob("*.txt"))
    if not raw_files:
        print(f"No .txt files found in {config.DATA_RAW}. Add your corpus first.")
        sys.exit(1)

    all_chunks: list[str] = []
    for path in raw_files:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        chunks = chunk_text(text, config.CHUNK_TOKENS, config.CHUNK_OVERLAP)
        print(f"{path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Defensive post-filter: drop any chunk that still carries license/credit noise
    NOISE = ("project gutenberg", "e-text prepared", "small print", "www.gutenberg")
    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if not any(n in c.lower() for n in NOISE)]
    if before - len(all_chunks):
        print(f"Dropped {before - len(all_chunks)} boilerplate-contaminated chunks")

    random.Random(config.SEED).shuffle(all_chunks)

    n = len(all_chunks)
    n_train = int(n * config.SPLITS[0])
    n_val = int(n * config.SPLITS[1])

    splits = {
        config.TRAIN_FILE: all_chunks[:n_train],
        config.VAL_FILE: all_chunks[n_train : n_train + n_val],
        config.HELDOUT_FILE: all_chunks[n_train + n_val :],
    }

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for out_path, chunks in splits.items():
        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(chunks):>5} chunks -> {out_path.name}")

    avg_tokens = sum(approx_token_count(c) for c in all_chunks) / max(1, n)
    print(f"\nTotal: {n} chunks | avg ~{avg_tokens:.0f} tokens/chunk | seed={config.SEED}")


if __name__ == "__main__":
    main()
