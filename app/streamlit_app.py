"""MyStyle Writer — base vs fine-tuned side-by-side comparison.

Two modes, auto-detected:

LIVE MODE   — if a LoRA adapter exists locally (models/adapters/...), loads the
              model and generates on the fly. Needs a GPU (or patience on CPU).
DEMO MODE   — if `evaluation/generated_base.jsonl` + `generated_tuned.jsonl`
              exist (produced on Colab), serves pre-generated pairs. Runs
              anywhere, including free Streamlit Cloud / HF Spaces.

Run:  streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
import config

BASE_GENS = ROOT / "evaluation" / "generated_base.jsonl"
TUNED_GENS = ROOT / "evaluation" / "generated_tuned.jsonl"
RESULTS = ROOT / "evaluation" / "results.json"

st.set_page_config(page_title="MyStyle Writer", page_icon="🖋️", layout="wide")
st.title("🖋️ MyStyle Writer — LoRA Style Fine-Tuning")
st.caption(f"Base model: `{config.BASE_MODEL}` · Style corpus: Oscar Wilde (public domain)")


# ---------------------------------------------------------------- helpers
def find_adapter() -> Path | None:
    adapters = sorted((ROOT / "models" / "adapters").glob("*/adapter_config.json"))
    return adapters[-1].parent if adapters else None


def load_demo_pairs():
    def read(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    base, tuned = read(BASE_GENS), read(TUNED_GENS)
    pairs = {}
    for b in base:
        pairs.setdefault(b["prompt"], {"base": [], "tuned": []})["base"].append(b["text"])
    for t in tuned:
        pairs.setdefault(t["prompt"], {"base": [], "tuned": []})["tuned"].append(t["text"])
    return pairs


@st.cache_resource(show_spinner="Loading model (first time takes a while)...")
def load_live_model(adapter_dir: str):
    from inference import StyleModel
    return StyleModel(adapter_dir=adapter_dir)


def render_pair(base_text: str, tuned_text: str):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base model")
        st.markdown(f"> {base_text}")
    with col2:
        st.subheader("Fine-tuned (Wilde LoRA)")
        st.markdown(f"> {tuned_text}")


# ---------------------------------------------------------------- sidebar
adapter = find_adapter()
demo_available = BASE_GENS.exists() and TUNED_GENS.exists()

with st.sidebar:
    st.header("Settings")
    if adapter:
        mode = st.radio("Mode", ["Live generation", "Pre-generated demo"] if demo_available else ["Live generation"])
    elif demo_available:
        mode = "Pre-generated demo"
        st.info("No local adapter found — running in demo mode with pre-generated outputs.")
    else:
        st.error("No adapter and no pre-generated outputs found. Train first (see notebooks/train_lora.ipynb).")
        st.stop()

    if mode == "Live generation":
        temperature = st.slider("Temperature", 0.1, 1.5, config.GEN_TEMPERATURE, 0.05)
        max_tokens = st.slider("Max new tokens", 64, 512, config.GEN_MAX_NEW_TOKENS, 32)

    if RESULTS.exists():
        st.divider()
        st.header("Evaluation")
        r = json.loads(RESULTS.read_text())
        emb = r["embedding_score"]
        st.metric("Style score — fine-tuned", f"{emb['tuned']['mean']:.4f}",
          delta=f"{r['delta_tuned_vs_base']:+.4f} vs base")
        st.caption("Cosine similarity to held-out Wilde centroid. Positive delta = model moved toward the target style.")


# ---------------------------------------------------------------- main
if mode == "Pre-generated demo":
    pairs = load_demo_pairs()
    prompt = st.selectbox("Choose a prompt", list(pairs.keys()))
    variants = pairs[prompt]
    n = min(len(variants["base"]), len(variants["tuned"]))
    idx = st.number_input("Sample", 1, max(1, n), 1) - 1
    render_pair(variants["base"][idx], variants["tuned"][idx])

else:  # Live generation
    from inference import EVAL_PROMPTS

    sm = load_live_model(str(adapter))
    preset = st.selectbox("Sample prompts", ["(write your own)"] + EVAL_PROMPTS)
    prompt = st.text_area("Prompt", value="" if preset == "(write your own)" else preset,
                          placeholder="Write a short passage about...")
    if st.button("Generate comparison", type="primary") and prompt.strip():
        with st.spinner("Generating from both models..."):
            base_text = sm.generate(prompt, use_adapter=False,
                                    temperature=temperature, max_new_tokens=max_tokens)
            tuned_text = sm.generate(prompt, use_adapter=True,
                                     temperature=temperature, max_new_tokens=max_tokens)
        render_pair(base_text, tuned_text)

st.divider()
st.caption(
    "How it works: a LoRA adapter (r=16) was fine-tuned on ~510-token chunks of "
    "Oscar Wilde's prose using 4-bit quantized Qwen2.5-1.5B on a free Colab T4. "
    "The evaluation scores generations against a held-out split never seen in training."
)
