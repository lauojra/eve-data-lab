from __future__ import annotations
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, ensure_dirs


def _market_schema() -> T.StructType:
    """
    Explicit schema = faster + consistent types across hundreds of CSV files.
    """
    return T.StructType([
        T.StructField("average", T.DoubleType(), True),
        T.StructField("date", T.StringType(), True),  
        T.StructField("highest", T.DoubleType(), True),
        T.StructField("lowest", T.DoubleType(), True),
        T.StructField("order_count", T.IntegerType(), True),
        T.StructField("volume", T.LongType(), True),
        T.StructField("http_last_modified", T.StringType(), True), 
        T.StructField("region_id", T.IntegerType(), True),
        T.StructField("type_id", T.IntegerType(), True),
    ])


def merge_clean_to_parquet(
    spark: SparkSession,
    input_glob: str | None = None,
    out_subdir: str = "market_history_parquet",
) -> None:
    """
    Read all daily CSV files, clean and standardize types, then write to Parquet.

    input_glob: optional override; default reads all CSV in data/interim.
    """
    ensure_dirs()

    # Get all CSV file paths
    csv_paths = glob.glob(str(INTERIM_DIR / "*.csv"))
    print(f">>> CSV files found: {len(csv_paths)}")
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {INTERIM_DIR}")

    out_dir = PROCESSED_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        spark.read
        .schema(_market_schema())
        .option("header", "true")
        .csv(csv_paths)
    )

    # Standardize types
    df = (
        df
        .withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        .withColumn("http_last_modified", F.to_timestamp("http_last_modified"))
    )

    # Remove rows with invalid / missing data
    df = df.filter(F.col("date").isNotNull())
    df = df.filter(F.col("type_id").isNotNull() & F.col("region_id").isNotNull())
    df = df.filter((F.col("average") >= 0) & (F.col("highest") >= 0) & (F.col("lowest") >= 0))
    df = df.filter(F.col("volume") >= 0)
    df = df.filter(F.col("highest") >= F.col("lowest"))

    # Remove duplicates 
    df = df.dropDuplicates(["date", "region_id", "type_id"])

    # Derived columns for easier partitioning/analysis
    df = (
        df
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )

    # Write Parquet (partitioned). Partitioning speeds up later reads by date.
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(str(out_dir))
    )

    print(">>> DONE: Parquet written.")
