from __future__ import annotations

from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

from src.utils.paths import PROCESSED_DIR, ensure_dirs


def build_type_stats(daily_df: DataFrame) -> DataFrame:
    """
    Build compact per-type statistics to cluster items.
    We aggregate across all regions for stability.
    """
    eps = F.lit(1e-6)

    df = daily_df.select(
        "type_id", "region_id", "average", "volume",
        *[c for c in ["highest", "lowest"] if c in daily_df.columns]
    )

    # Safe logs
    df = df.withColumn("log_price", F.log1p(F.col("average")))
    df = df.withColumn("log_volume", F.log1p(F.col("volume").cast("double")))

    # Spread if available
    if "highest" in df.columns and "lowest" in df.columns:
        df = df.withColumn("spread", (F.col("highest") - F.col("lowest")).cast("double"))
        df = df.withColumn("spread_rel", F.col("spread") / (F.col("average") + eps))
    else:
        df = df.withColumn("spread_rel", F.lit(0.0))

    stats = (
        df.groupBy("type_id")
        .agg(
            F.count("*").alias("n_rows"),
            F.countDistinct("region_id").alias("n_regions"),

            F.expr("percentile_approx(log_price, 0.5)").alias("log_price_med"),
            F.stddev("log_price").alias("log_price_std"),

            F.expr("percentile_approx(log_volume, 0.5)").alias("log_vol_med"),
            F.stddev("log_volume").alias("log_vol_std"),

            F.expr("percentile_approx(spread_rel, 0.5)").alias("spread_rel_med"),
        )
        .fillna(0.0, subset=["log_price_std", "log_vol_std"])
        .filter(F.col("n_rows") >= 200)  # quality filter, adjust if needed
    )

    return stats


def fit_kmeans_segments(
    spark: SparkSession,
    daily_df: DataFrame,
    k: int = 8,
    seed: int = 42,
    out_path: Path | None = None,
) -> DataFrame:
    """
    Fit KMeans on per-type stats and return a mapping: type_id -> segment_id.
    Also saves it to data/processed/type_segments_kmeans.
    """
    ensure_dirs()
    if out_path is None:
        out_path = PROCESSED_DIR / "type_segments_kmeans"

    daily_df = spark.read.parquet("data/processed/market_history_parquet")
    stats = build_type_stats(daily_df)

    feature_cols = [
        "log_price_med", "log_price_std",
        "log_vol_med", "log_vol_std",
        "spread_rel_med",
        "n_regions",
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    kmeans = KMeans(featuresCol="features", predictionCol="segment_id", k=k, seed=seed)

    pipe = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipe.fit(stats)

    seg = (
        model.transform(stats)
        .select("type_id", "segment_id", *feature_cols, "n_rows")
    )

    seg.write.mode("overwrite").parquet(str(out_path))
    print(f">>> Saved KMeans segments to: {out_path} (k={k})")

    return seg
