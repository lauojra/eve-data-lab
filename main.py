import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.download import download_market_history_years
from src.extract import extract_all 

YEARS_TO_FETCH = [2025]

def cmd_download(args: argparse.Namespace) -> None:
    years = args.years if args.years else YEARS_TO_FETCH
    print(f">>> Download years: {years}")
    download_market_history_years(years=years, overwrite=args.overwrite)


def cmd_extract(_args):
    extracted = extract_all()
    print(f"Extracted {len(extracted)} files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EVE Online - Market History pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_download = sub.add_parser("download", help="Download market-history archives for given years")
    p_download.add_argument("--years", nargs="*", type=int, help="Years to download, e.g. --years 2024 2025")
    p_download.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    p_download.set_defaults(func=cmd_download)

    p_extract = sub.add_parser("extract", help="Extract .gz/.bz2 archives into CSV")
    p_extract.set_defaults(func=cmd_extract)

    return parser

def main() -> None:
    print(">>> main.py starts")
    parser = build_parser()
    args = parser.parse_args()
    print(">>> parsed cmd:", args.cmd)
    args.func(args)

if __name__ == "__main__":
    main()