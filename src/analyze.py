from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.paths import PROCESSED_DIR, ensure_dirs

def run_eda(
    spark: SparkSession,
    parquet_subdir: str = "market_history_parquet",
    top_n_items: int = 20,
    sample_items_for_trends: int = 5,
) -> None:
    """
    EDA for EVE market history dataset stored as Parquet.

    Assumes dataset schema similar to:
      average, highest, lowest, order_count, volume, http_last_modified, region_id, type_id, date, year, month
    """
    ensure_dirs()
    path = PROCESSED_DIR / parquet_subdir
    print(f">>> EDA: reading Parquet from: {path}")

    df = spark.read.parquet(str(path)).cache()

    print("\n=== 1) Basic dataset overview ===")
    print(">>> row count:", df.count())
    df.select("date").agg(F.min("date").alias("min_date"), F.max("date").alias("max_date")).show(1, False)
    df.select(
        F.countDistinct("region_id").alias("n_regions"),
        F.countDistinct("type_id").alias("n_items"),
        F.countDistinct("date").alias("n_days"),
    ).show(1, False)

    print("\n=== 2) Schema & basic describe ===")
    df.printSchema()
    df.select("average", "highest", "lowest", "order_count", "volume").describe().show(20, False)

    print("\n=== 3) Missing values check (key columns) ===")
    missing = df.select(
        F.sum(F.col("date").isNull().cast("int")).alias("date_nulls"),
        F.sum(F.col("region_id").isNull().cast("int")).alias("region_nulls"),
        F.sum(F.col("type_id").isNull().cast("int")).alias("type_nulls"),
        F.sum(F.col("average").isNull().cast("int")).alias("average_nulls"),
        F.sum(F.col("volume").isNull().cast("int")).alias("volume_nulls"),
    )
    missing.show(1, False)

    print("\n=== 4) Sanity checks (invalid / suspicious rows) ===")
    sanity = df.select(
        F.sum((F.col("average") < 0).cast("int")).alias("avg_negative"),
        F.sum((F.col("highest") < 0).cast("int")).alias("highest_negative"),
        F.sum((F.col("lowest") < 0).cast("int")).alias("lowest_negative"),
        F.sum((F.col("volume") < 0).cast("int")).alias("volume_negative"),
        F.sum((F.col("highest") < F.col("lowest")).cast("int")).alias("highest_lt_lowest"),
    )
    sanity.show(1, False)

    print("\n=== 5) Derived quick features for EDA ===")
    dfe = (
        df.withColumn("spread", (F.col("highest") - F.col("lowest")).cast("double"))
          .withColumn("dow", F.date_format("date", "E"))  # Mon, Tue...
          .withColumn("dow_num", F.dayofweek("date"))      # 1=Sun ... 7=Sat (Spark)
    ).cache()

    print("\n=== 6) Top items by total volume (global) ===")
    top_items = (
        dfe.groupBy("type_id")
           .agg(
                F.sum("volume").alias("total_volume"),
                F.avg("average").alias("avg_price"),
                F.avg("spread").alias("avg_spread"),
                F.countDistinct("date").alias("days_present"),
           )
           .orderBy(F.desc("total_volume"))
           .limit(top_n_items)
    )
    top_items.show(top_n_items, False)

    print("\n=== 7) Top regions by total volume ===")
    top_regions = (
        dfe.groupBy("region_id")
           .agg(F.sum("volume").alias("total_volume"), F.countDistinct("type_id").alias("n_items"))
           .orderBy(F.desc("total_volume"))
           .limit(20)
    )
    top_regions.show(20, False)

    print("\n=== 8) Day-of-week seasonality (global) ===")
    # dow_num ensures correct ordering
    dow_stats = (
        dfe.groupBy("dow_num", "dow")
           .agg(
                F.avg("average").alias("avg_price"),
                F.avg("volume").alias("avg_volume"),
                F.avg("spread").alias("avg_spread"),
           )
           .orderBy("dow_num")
    )
    dow_stats.show(10, False)

    print("\n=== 9) Monthly seasonality (global) ===")
    # uses year/month columns if present; if not, derive from date
    if "year" in dfe.columns and "month" in dfe.columns:
        monthly = (
            dfe.groupBy("year", "month")
               .agg(
                    F.avg("average").alias("avg_price"),
                    F.sum("volume").alias("sum_volume"),
                    F.avg("spread").alias("avg_spread"),
               )
               .orderBy("year", "month")
        )
    else:
        monthly = (
            dfe.withColumn("year", F.year("date")).withColumn("month", F.month("date"))
               .groupBy("year", "month")
               .agg(
                    F.avg("average").alias("avg_price"),
                    F.sum("volume").alias("sum_volume"),
                    F.avg("spread").alias("avg_spread"),
               )
               .orderBy("year", "month")
        )
    monthly.show(24, False)

    print("\n=== 10) Volatility ranking (top by coeff of variation) ===")
    # CoV = std/mean, very common in price stability analysis
    vol_rank = (
        dfe.groupBy("type_id")
           .agg(
                F.avg("average").alias("mean_price"),
                F.stddev("average").alias("std_price"),
                F.sum("volume").alias("total_volume"),
           )
           .withColumn("cov", F.col("std_price") / F.col("mean_price"))
           .orderBy(F.desc("cov"))
           .limit(20)
    )
    vol_rank.show(20, False)

    print("\n=== 11) Correlations (global) ===")
    # Correlation requires numeric columns and non-null
    corr_df = dfe.select("average", "volume", "spread").dropna()
    print("corr(average, volume):", corr_df.stat.corr("average", "volume"))
    print("corr(average, spread):", corr_df.stat.corr("average", "spread"))
    print("corr(volume, spread):", corr_df.stat.corr("volume", "spread"))

    print("\n=== 12) Example price trends for a few top-volume items (global) ===")
    # Pick a few type_ids from top_items and show aggregated trend
    picked = [r["type_id"] for r in top_items.limit(sample_items_for_trends).collect()]
    print(">>> picked type_id:", picked)

    trends = (
        dfe.filter(F.col("type_id").isin(picked))
           .groupBy("date", "type_id")
           .agg(
                F.avg("average").alias("avg_price"),
                F.sum("volume").alias("sum_volume"),
                F.avg("spread").alias("avg_spread"),
           )
           .orderBy("type_id", "date")
    )
    trends.show(50, False)

    print("\n=== 13) Coverage check: missing days per item (quick proxy) ===")
    # Shows items with fewer observed days (incomplete series)
    coverage = (
        dfe.groupBy("type_id")
           .agg(F.countDistinct("date").alias("days_present"))
           .orderBy(F.asc("days_present"))
           .limit(20)
    )
    coverage.show(20, False)

    print("\n>>> EDA finished.\n")
    df.unpersist()
    dfe.unpersist()
