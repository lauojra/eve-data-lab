from pathlib import Path

# Project root directory (where main.py / requirements.txt live)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # compressed input files (.gz, .bz2)
INTERIM_DIR = DATA_DIR / "interim"  # decompressed CSV files
PROCESSED_DIR = DATA_DIR / "processed"  # parquet / final datasets


def ensure_dirs() -> None:
    """
    Ensure that all required data directories exist.
    This function is safe to call multiple times.
    """
    for p in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)