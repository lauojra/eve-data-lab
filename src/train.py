from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, FeatureHasher
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_dirs


@dataclass(frozen=True)
class SplitConfig:
    test_days: int = 60


@dataclass(frozen=True)
class TrainConfig:
    parquet_subdir: str = "training_dataset"
    segments_subdir: str = "type_segments_kmeans"   # where cmd_segment saved parquet
    label_col: str = "target_next_price"
    date_col: str = "date"
    region_col: str = "region_id"
    type_col: str = "type_id"


def _models_dir() -> Path:
    p = PROJECT_ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _eval(df_pred, label_col: str, pred_col: str = "prediction") -> dict[str, float]:
    rmse = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="rmse").evaluate(df_pred)
    mae = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="mae").evaluate(df_pred)
    r2 = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="r2").evaluate(df_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def _print(title: str, m: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    print(f"MAE : {m['mae']:.6f}")
    print(f"RMSE: {m['rmse']:.6f}")
    print(f"R2  : {m['r2']:.6f}")


def _time_split(df, date_col: str, test_days: int):
    max_date = df.select(F.max(F.col(date_col)).alias("max_date")).collect()[0]["max_date"]
    cutoff = F.date_sub(F.lit(max_date), test_days)
    train_df = df.filter(F.col(date_col) < cutoff)
    test_df = df.filter(F.col(date_col) >= cutoff)
    print(f">>> max_date={max_date} | cutoff(last {test_days} days)={cutoff}")
    print(f">>> train rows={train_df.count()} | test rows={test_df.count()}")
    return train_df, test_df


def _load_training_df(spark: SparkSession, cfg: TrainConfig):
    ensure_dirs()
    path = PROCESSED_DIR / cfg.parquet_subdir
    print(f">>> Loading training dataset: {path}")
    df = spark.read.parquet(str(path))

    # ensure string for hashing
    df = (
        df.withColumn(cfg.region_col, F.col(cfg.region_col).cast("string"))
          .withColumn(cfg.type_col, F.col(cfg.type_col).cast("string"))
    )

    df = df.dropna(subset=[cfg.label_col, cfg.date_col, cfg.region_col, cfg.type_col])
    return df


def _load_segments_df(spark: SparkSession, cfg: TrainConfig):
    path = PROCESSED_DIR / cfg.segments_subdir
    print(f">>> Loading segments mapping: {path}")
    seg = spark.read.parquet(str(path))

    # normalize types for join: our training df uses type_id as STRING (for hashing),
    # so we cast segment mapping as string too
    seg = seg.select(
        F.col("type_id").cast("string").alias(cfg.type_col),
        F.col("segment_id").cast("int").alias("segment_id"),
    ).dropna(subset=[cfg.type_col, "segment_id"])

    return seg


def _baseline_naive(df_any, label_col: str):
    if "price_lag_1" not in df_any.columns:
        raise ValueError("Missing 'price_lag_1' for baseline.")
    pred = df_any.select(F.col(label_col).alias(label_col), F.col("price_lag_1").alias("prediction")).dropna()
    return _eval(pred, label_col=label_col)


def _filter_top_items(df, cfg: TrainConfig, top_n_items: int):
    """
    Keep only top-N type_id by total volume to make training fast + meaningful.
    NOTE: type_id is STRING here.
    """
    if top_n_items <= 0:
        return df

    vol_col = "volume_lag_1" if "volume_lag_1" in df.columns else ("volume" if "volume" in df.columns else None)
    if vol_col is None:
        return df

    top = (
        df.groupBy(cfg.type_col)
          .agg(F.sum(F.col(vol_col)).alias("total_volume"))
          .orderBy(F.desc("total_volume"))
          .limit(top_n_items)
    )

    top_ids = [r[cfg.type_col] for r in top.collect()]
    print(f">>> Keeping top_n_items={top_n_items} by volume. Example ids: {top_ids[:5]}")
    return df.filter(F.col(cfg.type_col).isin(top_ids))


def train_ridge_per_segment(
    spark: SparkSession,
    split_cfg: SplitConfig = SplitConfig(),
    cfg: TrainConfig = TrainConfig(),
    # speed knobs
    top_n_items: int = 200,
    sample_fraction: float = 0.2,
    seed: int = 42,
    # segmentation knobs
    top_n_segments: int = 5,
    min_rows_train: int = 5_000,
    min_rows_test: int = 1_000,
    # hashing knobs
    hash_num_features: int = 2**15,
    # ridge knobs
    reg_param: float = 0.1,
    elastic_net_param: float = 0.0,
    max_iter: int = 50,
) -> None:
    """
    Train SAME Ridge pipeline as in your original file, but separately per segment_id.

    Output:
      models/ridge_segmented/segment_id=<id>/
    """
    df = _load_training_df(spark, cfg)
    seg = _load_segments_df(spark, cfg)

    # join segments
    df = df.join(seg, on=cfg.type_col, how="inner")
    print(f">>> After join with segments: rows={df.count()}")

    # Optional: keep only most traded items
    df = _filter_top_items(df, cfg, top_n_items=top_n_items)

    # Optional: sample rows for speed
    if 0 < sample_fraction < 1.0:
        print(f">>> Sampling fraction={sample_fraction} (seed={seed})")
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    df = df.cache()

    # pick largest segments to keep runtime predictable
    seg_ids = (
        df.groupBy("segment_id").count()
          .orderBy(F.desc("count"))
          .limit(top_n_segments)
          .select("segment_id")
          .collect()
    )
    seg_ids = [r["segment_id"] for r in seg_ids]
    print(f">>> Training top_n_segments={top_n_segments}: {seg_ids}")

    # numeric features
    numeric_cols = [
        "price_lag_1",
        "price_lag_3",
        "price_lag_7",
        "price_mean_7",
        "price_mean_14",
        "price_std_7",
        "volume_lag_1",
        "volume_mean_7",
        "spread",
        "spread_mean_7",
        "dow",
        "month",
    ]

    assembler_num = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_vec", handleInvalid="keep")

    hasher = FeatureHasher(
        inputCols=[cfg.region_col, cfg.type_col],
        outputCol="cat_vec",
        numFeatures=int(hash_num_features),
    )

    assembler_all = VectorAssembler(
        inputCols=["numeric_vec", "cat_vec"],
        outputCol="features_raw",
        handleInvalid="keep",
    )

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    ridge = LinearRegression(
        featuresCol="features",
        labelCol=cfg.label_col,
        predictionCol="prediction",
        elasticNetParam=float(elastic_net_param),
        regParam=float(reg_param),
        maxIter=int(max_iter),
    )

    ridge_pipe = Pipeline(stages=[assembler_num, hasher, assembler_all, scaler, ridge])

    out_root = _models_dir() / "ridge_segmented"
    out_root.mkdir(parents=True, exist_ok=True)

    # loop per segment
    for seg_id in seg_ids:
        print(f"\n=============================")
        print(f">>> SEGMENT {seg_id}")
        print(f"=============================")

        seg_df = df.filter(F.col("segment_id") == F.lit(seg_id)).cache()

        train_df, test_df = _time_split(seg_df, cfg.date_col, split_cfg.test_days)
        train_df = train_df.cache()
        test_df = test_df.cache()

        train_n = train_df.count()
        test_n = test_df.count()
        print(f">>> segment rows: train={train_n} test={test_n}")

        if train_n < min_rows_train or test_n < min_rows_test:
            print(f">>> Skipping segment {seg_id} (too few rows).")
            seg_df.unpersist()
            train_df.unpersist()
            test_df.unpersist()
            continue

        print("\n>>> Baseline (naive)")
        base_train = _baseline_naive(train_df, cfg.label_col)
        _print(f"Baseline (TRAIN) seg={seg_id}", base_train)

        base_test = _baseline_naive(test_df, cfg.label_col)
        _print(f"Baseline (TEST)  seg={seg_id}", base_test)

        print("\n>>> Training Ridge (fast pipeline) ...")
        model = ridge_pipe.fit(train_df)

        # TRAIN metrics
        pred_train = model.transform(train_df).select(cfg.label_col, "prediction").dropna()
        m_train = _eval(pred_train, cfg.label_col)
        _print(f"Ridge (TRAIN) seg={seg_id}", m_train)

        # TEST metrics
        pred_test = model.transform(test_df).select(cfg.label_col, "prediction").dropna()
        m_test = _eval(pred_test, cfg.label_col)
        _print(f"Ridge (TEST)  seg={seg_id}", m_test)

        print("\n=== SUMMARY (TEST) ===")
        print(f"seg={seg_id} | Baseline MAE={base_test['mae']:.4f} RMSE={base_test['rmse']:.4f} R2={base_test['r2']:.4f}")
        print(f"seg={seg_id} | Ridge    MAE={m_test['mae']:.4f} RMSE={m_test['rmse']:.4f} R2={m_test['r2']:.4f}")

        seg_out = out_root / f"segment_id={seg_id}"
        model.write().overwrite().save(str(seg_out))
        print(f">>> Saved Ridge model: {seg_out}")

        # cleanup per segment
        seg_df.unpersist()
        train_df.unpersist()
        test_df.unpersist()

    df.unpersist()
