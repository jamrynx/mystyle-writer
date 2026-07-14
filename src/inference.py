"""Unified inference: generate from the base model and from base+LoRA adapter.

Both paths use identical sampling parameters so comparisons are fair.

Interactive use:
    from inference import StyleModel
    sm = StyleModel(adapter_dir="models/adapters/wilde-full")
    sm.generate("Write about a garden at dusk.", use_adapter=True)

Batch mode (feeds the evaluation — run this on Colab after training):
    python src/inference.py --adapter models/adapters/wilde-full \
        --out-base evaluation/generated_base.jsonl \
        --out-tuned evaluation/generated_tuned.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(str(Path(__file__).resolve().parent))
import config

# Prompts used for batch generation and evaluation.
# Deliberately style-neutral so any Wilde-ness in output comes from the model.
EVAL_PROMPTS = [
    "Write a short passage about a stranger arriving in a quiet town.",
    "Describe an elegant dinner party from the perspective of a bored guest.",
    "Write a paragraph about the relationship between beauty and morality.",
    "Describe an old portrait hanging in a dim hallway.",
    "Write a short exchange between two friends discussing marriage.",
    "Describe a garden in the early evening.",
    "Write a passage about a secret that weighs on someone's conscience.",
    "Describe a fashionable London street on a summer afternoon.",
    "Write a paragraph about the price of pleasure.",
    "Describe a ghost who is tired of haunting.",
]


class StyleModel:
    def __init__(self, adapter_dir: str | None = None, load_in_4bit: bool | None = None):
        if load_in_4bit is None:
            load_in_4bit = torch.cuda.is_available() and config.LOAD_IN_4BIT

        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif not torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float32  # CPU fallback

        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **kwargs)

        self.has_adapter = False
        if adapter_dir and Path(adapter_dir).exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.has_adapter = True

        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, use_adapter: bool, max_new_tokens: int | None = None,
                 temperature: float | None = None, seed: int | None = None) -> str:
        if use_adapter and not self.has_adapter:
            raise RuntimeError("No adapter loaded — pass adapter_dir to StyleModel.")
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", add_generation_prompt=True,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or config.GEN_MAX_NEW_TOKENS,
            temperature=temperature or config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if self.has_adapter and not use_adapter:
            with self.model.disable_adapter():
                out = self.model.generate(inputs, **gen_kwargs)
        else:
            out = self.model.generate(inputs, **gen_kwargs)

        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


STYLE_INSTRUCTION = (
    "Write in the style of Oscar Wilde: ornate, epigrammatic Victorian prose "
    "with wit and paradox. "
)


def batch_generate(adapter_dir: str, out_base: str, out_tuned: str,
                   out_prompted: str | None = None,
                   samples_per_prompt: int = 2) -> None:
    """Generate outputs for every eval prompt under three conditions:
    base, style-prompted base (the 'just prompt it' alternative), and fine-tuned.
    This makes the evaluation an empirical answer to the fine-tune-vs-prompt
    decision framework (Unit 3)."""
    sm = StyleModel(adapter_dir=adapter_dir)
    conditions = [(out_base, False, ""), (out_tuned, True, "")]
    if out_prompted:
        conditions.insert(1, (out_prompted, False, STYLE_INSTRUCTION))
    for out_path, use_adapter, prefix in conditions:
        rows = []
        for prompt in EVAL_PROMPTS:
            for s in range(samples_per_prompt):
                text = sm.generate(prefix + prompt, use_adapter=use_adapter, seed=config.SEED + s)
                rows.append({"prompt": prompt, "text": text})
                label = "tuned" if use_adapter else ("prompted" if prefix else "base")
                print(f"[{label:>8}] {prompt[:50]}... ({s+1})")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} generations -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    ap.add_argument("--out-base", default="evaluation/generated_base.jsonl")
    ap.add_argument("--out-tuned", default="evaluation/generated_tuned.jsonl")
    ap.add_argument("--out-prompted", default="evaluation/generated_prompted.jsonl")
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    args = ap.parse_args()
    batch_generate(args.adapter, args.out_base, args.out_tuned,
                   args.out_prompted, args.samples_per_prompt)
