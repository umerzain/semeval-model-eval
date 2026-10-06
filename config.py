"""Edit MODEL_KEY to switch checkpoints; the remaining settings are shared."""

# MODEL_KEY = "internvl3_5_4b"
MODEL_KEY = "CulturalPangea-7B"

# Select models on labeled dev, matching the released evaluation-style inputs.
SPLITS = ("dev",)  # Add "train" only for later analysis or training.
TRACKS = "all"  # or e.g. ("qa_mena_en", "qa_mena_arz")

# Keep the benchmark input unchanged. Add "jpeg_85" only for a separate probe.
VARIANTS = ("original",)
ROBUSTNESS_SPLITS = ("dev",)

OUTPUT_DIR = "/kaggle/working/mmcqa_runs"
DATA_DIR = "/kaggle/working/mmcqa_data"
MAX_ROWS_PER_TRACK = 100  # Set to 5 for a smoke run; None runs every row.
MAX_NEW_TOKENS = 128
VISUAL_MAX_NEW_TOKENS = 512
LOAD_IN_4BIT = True
SEED = 42
BERTSCORE_BATCH_SIZE = 8
BERTSCORE_DEVICE = None  # Auto-select GPU when available.
BERTSCORE_MODEL = "bert-base-multilingual-cased"  # Keep the same scorer for local comparisons.
