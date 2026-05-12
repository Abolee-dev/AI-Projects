import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "sqlrag.db"))
DATASET_REGISTRY_PATH = os.getenv(
    "DATASET_REGISTRY_PATH",
    str(ROOT_DIR / "config" / "dataset_registry.yaml")
)
DEFAULT_ROW_LIMIT = int(os.getenv("DEFAULT_ROW_LIMIT", "100"))
