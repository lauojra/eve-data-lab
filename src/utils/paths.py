from pathlib import Path

# Project root directory (where main.py / requirements.txt live)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        
INTERIM_DIR = DATA_DIR / "interim"  
PROCESSED_DIR = DATA_DIR / "processed" 


def ensure_dirs() -> None:
    """
    Ensure that all required data directories exist.
    This function is safe to call multiple times.
    """
    for p in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)