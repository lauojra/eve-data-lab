import sys
import argparse
from pathlib import Path
from pyspark.sql import SparkSession
# from torch import sub

sys.path.append(str(Path(__file__).parent / "src"))

from src.download import download_market_history_years
from src.extract import extract_all 
from src.transform import merge_clean_to_parquet

YEARS_TO_FETCH = [2025]

def _make_spark(app_name: str):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

def cmd_download(args: argparse.Namespace) -> None:
    years = args.years if args.years else YEARS_TO_FETCH
    print(f">>> Download years: {years}")
    download_market_history_years(years=years, overwrite=args.overwrite)

def cmd_extract(_args):
    extracted = extract_all()
    print(f"Extracted {len(extracted)} files")

def cmd_transform(args):
    spark = _make_spark("eve-transform")
    try:
        merge_clean_to_parquet(spark)
    finally:
        spark.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EVE Online - Market History pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_download = sub.add_parser("download", help="Download market-history archives for given years")
    p_download.add_argument("--years", nargs="*", type=int, help="Years to download, e.g. --years 2024 2025")
    p_download.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    p_download.set_defaults(func=cmd_download)

    p_extract = sub.add_parser("extract", help="Extract .gz/.bz2 archives into CSV")
    p_extract.set_defaults(func=cmd_extract)

    p_transform = sub.add_parser("transform", help="Transform CSV into Parquet")
    p_transform.set_defaults(func=cmd_transform)

    return parser

def main() -> None:

    parser = build_parser()
    args = parser.parse_args()
    print(">>> parsed cmd:", args.cmd)
    args.func(args)

if __name__ == "__main__":
    main()