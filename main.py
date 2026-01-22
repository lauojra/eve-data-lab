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
from src.segment import fit_kmeans_segments
from src.train import train_ridge_per_segment

YEARS_TO_FETCH = [2025]

def _make_spark(app_name: str):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
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
    spark.sparkContext.setLogLevel("ERROR")
    try:
        merge_clean_to_parquet(spark)
    finally:
        spark.stop()

def cmd_analyze(args):
    spark = _make_spark("eve-analyze")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        run_eda(spark)
    finally:
        spark.stop()

def cmd_plot(args):
    spark = _make_spark("eve-plots")
    spark.sparkContext.setLogLevel("ERROR")
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
    spark.sparkContext.setLogLevel("ERROR")
    try:
        build_feature_dataset(spark)
    finally:
        spark.stop()

def cmd_segment(args):
    spark = _make_spark("eve-segment")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        fit_kmeans_segments(spark, k=args.k)
    finally:
        spark.stop()    


def cmd_train_segmented_ridge(args):
    spark = _make_spark("eve-train-segmented")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        train_ridge_per_segment(
            spark,
            # knobs
            top_n_items=args.top_n_items,
            sample_fraction=args.sample_fraction,
            top_n_segments=args.top_n_segments,
            reg_param=args.reg_param,
            elastic_net_param=args.elastic_net,
        )
    finally:
        spark.stop()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EVE Online - Market History pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) Download data from EVE Online
    p_download = sub.add_parser("download", help="Download market-history archives for given years")
    p_download.add_argument("--years", nargs="*", type=int, help="Years to download, e.g. --years 2024 2025")
    p_download.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    p_download.set_defaults(func=cmd_download)

    # 2) Data extraction from .gz/.bz2 to CSV
    p_extract = sub.add_parser("extract", help="Extract .gz/.bz2 archives into CSV")
    p_extract.set_defaults(func=cmd_extract)

    # 3) Data transformation
    p_transform = sub.add_parser("transform", help="Transform CSV into Parquet")
    p_transform.set_defaults(func=cmd_transform)

    # 4) Analyze / EDA
    p_analyze = sub.add_parser("analyze", help="Analyze market history data")
    p_analyze.set_defaults(func=cmd_analyze)

    # 5) Visualizations / plots
    p_plot = sub.add_parser("plot", help="Generate plots (saved to reports/figures)")
    p_plot.add_argument("--mode", choices=["lines", "subplots", "scatter", "spread"], default="lines")
    p_plot.add_argument("--type-id", type=int, required=True)
    p_plot.add_argument("--regions", nargs="+", type=int, required=True, help="Region IDs (1+). For scatter/spread use one region.")
    p_plot.add_argument("--date-from", type=str, default=None, help="YYYY-MM-DD")
    p_plot.add_argument("--date-to", type=str, default=None, help="YYYY-MM-DD")
    p_plot.set_defaults(func=cmd_plot)
    
    # 6) Feature engineering
    p_feat = sub.add_parser("features", help="Build ML feature dataset")
    p_feat.set_defaults(func=cmd_features)

    # 7) Segmentation with KMeans clustering
    p_seg = sub.add_parser("segment", help="KMeans segmentation (type_id) + Ridge/ElasticNet per segment")
    p_seg.add_argument("--k", type=int, default=8, help="Number of KMeans clusters")
    p_seg.set_defaults(func=cmd_segment)

    # 8) Regression per segment with Ridge/ElasticNet
    p_reg = sub.add_parser("train-segmented-ridge", help="Train Ridge per segment_id (requires cmd_segment output)")
    p_reg.add_argument("--top-n-items", type=int, default=200)
    p_reg.add_argument("--sample-fraction", type=float, default=0.2)
    p_reg.add_argument("--top-n-segments", type=int, default=5)
    p_reg.add_argument("--reg-param", type=float, default=0.1)
    p_reg.add_argument("--elastic-net", type=float, default=0.0)  # 0.0 Ridge, 0.5 ElasticNet
    p_reg.set_defaults(func=cmd_train_segmented_ridge)

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(">>> parsed cmd:", args.cmd)
    args.func(args)

if __name__ == "__main__":
    main()