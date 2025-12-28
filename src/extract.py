import gzip
import bz2
import shutil
from pathlib import Path

from utils.paths import RAW_DIR, INTERIM_DIR, ensure_dirs


def _strip_known_compression_suffix(filename: str) -> str:
    """
    Remove known compression suffixes from a filename.

    Examples:
    - market-history.csv.gz  -> market-history.csv
    - market-history.csv.bz2 -> market-history.csv
    """
    if filename.endswith(".gz"):
        return filename[:-3]
    if filename.endswith(".bz2"):
        return filename[:-4]
    return filename


def decompress_file(input_path: Path) -> Path:
    """
    Decompress a single .gz or .bz2 file into the interim directory.

    The file is streamed (not fully loaded into memory),
    which makes it safe for large files.

    Returns:
        Path to the decompressed output file.
    """
    ensure_dirs()

    suffix = input_path.suffix.lower()
    if suffix not in [".gz", ".bz2"]:
        raise ValueError(
            f"Unsupported file type: {input_path} (expected .gz or .bz2)"
        )

    # Determine output filename
    output_name = _strip_known_compression_suffix(input_path.name)
    output_path = INTERIM_DIR / output_name

    print(f">>> Decompressing: {input_path.name} -> {output_path.name}")

    # Select the correct decompression method
    opener = gzip.open if suffix == ".gz" else bz2.open

    # Stream copy: compressed file -> decompressed file
    with opener(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return output_path


def extract_all() -> list[Path]:
    """
    Find all compressed files in RAW_DIR (.gz and .bz2),
    decompress them into INTERIM_DIR,
    and return a list of decompressed file paths.
    """
    ensure_dirs()
    print(">>> extract_all() CALLED")

    # Collect all supported compressed files
    compressed_files = list(RAW_DIR.glob("*.gz")) + list(RAW_DIR.glob("*.bz2"))
    print(f">>> Found compressed files: {len(compressed_files)}")

    extracted_files: list[Path] = []
    for file_path in compressed_files:
        extracted_files.append(decompress_file(file_path))

    print(f">>> Successfully extracted files: {len(extracted_files)}")
    return extracted_files
