import sys
import argparse
from pathlib import Path
from pyspark.sql import SparkSession

sys.path.append(str(Path(__file__).parent / "src"))

from src.analyze import run_eda
from src.download import download_market_history_years
from src.extract import extract_all 
from src.transform import merge_clean_to_parquet
from src.plots import (
    plot_price_lines_by_region,
    plot_price_subplots_by_region,
    plot_price_vs_volume_scatter,
    plot_spread_over_time,
)
from src.features import build_feature_dataset



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

def cmd_extract(args):
    extracted = extract_all()
    print(f"Extracted {len(extracted)} files")

def cmd_transform(args):
    spark = _make_spark("eve-transform")
    try:
        merge_clean_to_parquet(spark)
    finally:
        spark.stop()

def cmd_analyze(args):
    spark = _make_spark("eve-analyze")
    try:
        run_eda(spark)
    finally:
        spark.stop()

def cmd_plot(args):
    spark = _make_spark("eve-plots")
    try:
        if args.mode == "lines":
            out = plot_price_lines_by_region(
                spark,
                type_id=args.type_id,
                region_ids=args.regions,
                date_from=args.date_from,
                date_to=args.date_to,
            )
        elif args.mode == "subplots":
            out = plot_price_subplots_by_region(
                spark,
                type_id=args.type_id,
                region_ids=args.regions,
                date_from=args.date_from,
                date_to=args.date_to,
            )
        elif args.mode == "scatter":
            out = plot_price_vs_volume_scatter(
                spark,
                type_id=args.type_id,
                region_id=args.regions[0],
                date_from=args.date_from,
                date_to=args.date_to,
            )
        elif args.mode == "spread":
            out = plot_spread_over_time(
                spark,
                type_id=args.type_id,
                region_id=args.regions[0],
                date_from=args.date_from,
                date_to=args.date_to,
            )
        else:
            raise ValueError("Unknown plot mode")

        print(f">>> Saved plot: {out}")
    finally:
        spark.stop()

def cmd_features(args):
    spark = _make_spark("eve-features")
    try:
        build_feature_dataset(spark)
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

    p_analyze = sub.add_parser("analyze", help="Analyze market history data")
    p_analyze.set_defaults(func=cmd_analyze)

    p_plot = sub.add_parser("plot", help="Generate plots (saved to reports/figures)")
    p_plot.add_argument("--mode", choices=["lines", "subplots", "scatter", "spread"], default="lines")
    p_plot.add_argument("--type-id", type=int, required=True)
    p_plot.add_argument("--regions", nargs="+", type=int, required=True, help="Region IDs (1+). For scatter/spread use one region.")
    p_plot.add_argument("--date-from", type=str, default=None, help="YYYY-MM-DD")
    p_plot.add_argument("--date-to", type=str, default=None, help="YYYY-MM-DD")
    p_plot.set_defaults(func=cmd_plot)
    
    p_feat = sub.add_parser("features", help="Build ML feature dataset")
    p_feat.set_defaults(func=cmd_features)

    return parser

def main() -> None:

    parser = build_parser()
    args = parser.parse_args()
    print(">>> parsed cmd:", args.cmd)
    args.func(args)

if __name__ == "__main__":
    main()