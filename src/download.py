from urllib.parse import urljoin
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from src.utils.paths import RAW_DIR, ensure_dirs

BASE_URL = "https://data.everef.net/market-history"


@dataclass(frozen=True)
class DownloadResult:
    year: int
    downloaded: list[Path]
    skipped_existing: int


def _list_year_files(year: int) -> list[str]:
    """
    Return absolute file URLs for a given year directory.
    """
    base_url = f"{BASE_URL}/{year}/"
    print(f">>> Fetching file list from: {base_url}")

    r = requests.get(base_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    file_urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Filter only files we care about
        if not href.endswith((".csv.gz", ".csv.bz2", ".gz", ".bz2")):
            continue

        # This safely builds a correct absolute URL
        full_url = urljoin(base_url, href)
        file_urls.append(full_url)

    return file_urls


def download_market_history_years(years: Iterable[int], overwrite: bool = False) -> list[DownloadResult]:
    """
    Download all market-history files for given years into data/raw.
    """
    ensure_dirs()
    results: list[DownloadResult] = []

    for year in years:
        print(f">>> Listing files for year: {year}")
        urls = _list_year_files(year)
        print(f">>> Found {len(urls)} files")

        downloaded: list[Path] = []
        skipped = 0

        for file_url in urls:
            filename = file_url.split("/")[-1]
            out_path = RAW_DIR / filename

            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            print(f">>> Downloading: {filename}")
            with requests.get(file_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            downloaded.append(out_path)

        results.append(DownloadResult(year=year, downloaded=downloaded, skipped_existing=skipped))
        print(f">>> Year {year}: downloaded={len(downloaded)}, skipped_existing={skipped}")

    return results
