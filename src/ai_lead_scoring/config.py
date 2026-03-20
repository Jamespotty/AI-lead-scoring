"""Configuration constants for AI Lead Scoring."""

OPENAI_MODEL = "gpt-4o-mini"
SCORE_BATCH_SIZE = 30
MAX_WORKERS = 8
TEXT_FIELD_LIMIT = 400
CHECKPOINT_EVERY = 10   # batches between disk flushes
CHECKPOINT_DIR = "."
