from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.paths import PROCESSED_DIR, ensure_dirs


def build_feature_dataset(
    spark: SparkSession,
    parquet_subdir: str = "market_history_parquet",
    out_subdir: str = "training_dataset",
) -> None:
    """
    Build ML-ready feature dataset for next-day average price prediction.
    """

    ensure_dirs()

    in_path = PROCESSED_DIR / parquet_subdir
    out_path = PROCESSED_DIR / out_subdir

    print(f">>> Reading input Parquet: {in_path}")
    df = spark.read.parquet(str(in_path))

    # Basic derived columns
    df = (
        df.withColumn("spread", (F.col("highest") - F.col("lowest")).cast("double"))
          .withColumn("dow", F.dayofweek("date"))
          .withColumn("month", F.month("date"))
          .withColumn("year", F.year("date"))
    )

    # Window per time series
    w = Window.partitionBy("region_id", "type_id").orderBy("date")

    # Lags
    df = (
        df.withColumn("price_lag_1", F.lag("average", 1).over(w))
          .withColumn("price_lag_3", F.lag("average", 3).over(w))
          .withColumn("price_lag_7", F.lag("average", 7).over(w))
          .withColumn("volume_lag_1", F.lag("volume", 1).over(w))
    )

    # Rolling windows (exclude current row to avoid leakage)
    w7 = w.rowsBetween(-7, -1)
    w14 = w.rowsBetween(-14, -1)

    df = (
        df.withColumn("price_mean_7", F.avg("average").over(w7))
          .withColumn("price_mean_14", F.avg("average").over(w14))
          .withColumn("price_std_7", F.stddev("average").over(w7))
          .withColumn("volume_mean_7", F.avg("volume").over(w7))
          .withColumn("spread_mean_7", F.avg("spread").over(w7))
    )

    # Target: next day price
    df = df.withColumn("target_next_price", F.lead("average", 1).over(w))

    # Remove rows with incomplete history
    feature_cols = [
        "price_lag_1", "price_lag_3", "price_lag_7",
        "price_mean_7", "price_mean_14", "price_std_7",
        "volume_lag_1", "volume_mean_7",
        "spread", "spread_mean_7", "dow",
        "target_next_price",
    ]

    df = df.dropna(subset=feature_cols)
    
    # Select final dataset
    final_df = df.select(
        "date",
        "year",
        "month",
        "region_id",
        "type_id",
        "average",
        *feature_cols
    )

    print(">>> Final feature dataset sample:")
    final_df.show(5, False)

    print(f">>> Writing training dataset to: {out_path}")

    (
        final_df.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(str(out_path))
    )

    print(">>> Feature engineering finished.")
