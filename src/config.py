"""Single source of truth for model, data, and training parameters.

Every script (prepare_data, training notebook, inference, evaluation)
imports from here so a change in one place propagates everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ADAPTERS_DIR = ROOT / "models" / "adapters"

TRAIN_FILE = DATA_PROCESSED / "train.jsonl"
VAL_FILE = DATA_PROCESSED / "val.jsonl"
HELDOUT_FILE = DATA_PROCESSED / "heldout.jsonl"  # eval-only, never trained on

# ---------------------------------------------------------------- model
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Fallback if Colab VRAM is tight or downloads are slow:
# BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---------------------------------------------------------------- data prep
CHUNK_TOKENS = 512          # approx tokens per training passage
CHUNK_OVERLAP = 64          # sliding-window overlap between chunks
SPLITS = (0.80, 0.10, 0.10) # train / val / heldout
SEED = 42

# ---------------------------------------------------------------- LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------- training
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2              # per-device, T4-friendly
GRAD_ACCUM = 8              # effective batch = 16
MAX_SEQ_LEN = 512
LOAD_IN_4BIT = True

# ---------------------------------------------------------------- generation
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.95
