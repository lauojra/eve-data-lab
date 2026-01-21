from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT


def _fig_dir() -> Path:
    out = PROJECT_ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_item_regions_timeseries(
    spark: SparkSession,
    parquet_subdir: str,
    type_id: int,
    region_ids: list[int],
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Load time series for one type_id across multiple regions and return as pandas DataFrame.
    Intended for plotting (small subset only).
    """
    path = PROCESSED_DIR / parquet_subdir
    df = spark.read.parquet(str(path))

    df = df.filter(F.col("type_id") == F.lit(type_id))
    df = df.filter(F.col("region_id").isin([int(x) for x in region_ids]))

    if date_from:
        df = df.filter(F.col("date") >= F.to_date(F.lit(date_from)))
    if date_to:
        df = df.filter(F.col("date") <= F.to_date(F.lit(date_to)))

    # Aggregate per day/region (just in case there are duplicates)
    df = (
        df.groupBy("date", "region_id")
          .agg(
              F.avg("average").alias("avg_price"),
              F.sum("volume").alias("sum_volume"),
              F.avg((F.col("highest") - F.col("lowest"))).alias("avg_spread"),
          )
          .orderBy("date", "region_id")
    )

    pdf = df.toPandas()
    if not pdf.empty:
        pdf["date"] = pd.to_datetime(pdf["date"])
    return pdf


def plot_price_lines_by_region(
    spark: SparkSession,
    type_id: int,
    region_ids: list[int],
    parquet_subdir: str = "market_history_parquet",
    date_from: str | None = None,
    date_to: str | None = None,
) -> Path:
    """
    One plot, multiple lines: price over time for one item across multiple regions.
    """
    pdf = _load_item_regions_timeseries(
        spark=spark,
        parquet_subdir=parquet_subdir,
        type_id=type_id,
        region_ids=region_ids,
        date_from=date_from,
        date_to=date_to,
    )

    out_path = _fig_dir() / f"price_lines_type_{type_id}_regions_{'-'.join(map(str, region_ids))}.png"

    plt.figure()
    if pdf.empty:
        plt.title(f"type_id={type_id} (no data for selected regions)")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        return out_path

    for rid in sorted(pdf["region_id"].unique()):
        sub = pdf[pdf["region_id"] == rid].sort_values("date")
        plt.plot(sub["date"], sub["avg_price"], label=f"region {rid}")

    plt.title(f"Average price over time | type_id={type_id}")
    plt.xlabel("date")
    plt.ylabel("avg_price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_price_subplots_by_region(
    spark: SparkSession,
    type_id: int,
    region_ids: list[int],
    parquet_subdir: str = "market_history_parquet",
    date_from: str | None = None,
    date_to: str | None = None,
) -> Path:
    """
    Subplots: one subplot per region for the same type_id.
    """
    pdf = _load_item_regions_timeseries(
        spark=spark,
        parquet_subdir=parquet_subdir,
        type_id=type_id,
        region_ids=region_ids,
        date_from=date_from,
        date_to=date_to,
    )

    out_path = _fig_dir() / f"price_subplots_type_{type_id}_regions_{'-'.join(map(str, region_ids))}.png"

    n = len(region_ids)
    n = max(n, 1)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 2.6 * n), sharex=True)

    if n == 1:
        axes = [axes]

    if pdf.empty:
        axes[0].set_title(f"type_id={type_id} (no data for selected regions)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path

    for ax, rid in zip(axes, region_ids):
        sub = pdf[pdf["region_id"] == rid].sort_values("date")
        ax.plot(sub["date"], sub["avg_price"])
        ax.set_title(f"region {rid}")
        ax.set_ylabel("avg_price")

    axes[-1].set_xlabel("date")
    fig.suptitle(f"Average price over time | type_id={type_id}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_price_vs_volume_scatter(
    spark: SparkSession,
    type_id: int,
    region_id: int,
    parquet_subdir: str = "market_history_parquet",
    date_from: str | None = None,
    date_to: str | None = None,
) -> Path:
    """
    Scatter: avg_price vs volume for a single item in a single region.
    Useful to show demand/price relationship.
    """
    pdf = _load_item_regions_timeseries(
        spark=spark,
        parquet_subdir=parquet_subdir,
        type_id=type_id,
        region_ids=[region_id],
        date_from=date_from,
        date_to=date_to,
    )

    out_path = _fig_dir() / f"scatter_price_vs_volume_type_{type_id}_region_{region_id}.png"

    plt.figure()
    if pdf.empty:
        plt.title(f"type_id={type_id}, region={region_id} (no data)")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        return out_path

    plt.scatter(pdf["sum_volume"], pdf["avg_price"], s=8)
    plt.title(f"Price vs Volume | type_id={type_id}, region={region_id}")
    plt.xlabel("sum_volume (per day)")
    plt.ylabel("avg_price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_spread_over_time(
    spark: SparkSession,
    type_id: int,
    region_id: int,
    parquet_subdir: str = "market_history_parquet",
    date_from: str | None = None,
    date_to: str | None = None,
) -> Path:
    """
    Spread over time (highest-lowest). This is a nice market liquidity proxy.
    """
    pdf = _load_item_regions_timeseries(
        spark=spark,
        parquet_subdir=parquet_subdir,
        type_id=type_id,
        region_ids=[region_id],
        date_from=date_from,
        date_to=date_to,
    )

    out_path = _fig_dir() / f"spread_over_time_type_{type_id}_region_{region_id}.png"

    plt.figure()
    if pdf.empty:
        plt.title(f"type_id={type_id}, region={region_id} (no data)")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        return out_path

    pdf = pdf.sort_values("date")
    plt.plot(pdf["date"], pdf["avg_spread"])
    plt.title(f"Average spread over time | type_id={type_id}, region={region_id}")
    plt.xlabel("date")
    plt.ylabel("avg_spread (highest-lowest)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path
