from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

from fnews.data.fakeddit import normalize_bool_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Fakeddit images from TSV image_url values.")
    parser.add_argument("--tsv", required=True, help="Path to a Fakeddit TSV file.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/fakeddit/images",
        help="Directory where images should be saved by submission id.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum rows to attempt.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout in seconds.")
    return parser.parse_args()


def infer_extension(url: str, content_type: str | None) -> str:
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if path_suffix == ".jpeg" else path_suffix

    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    return ".jpg"


def existing_image(output_dir: Path, image_id: str) -> Path | None:
    for suffix in (".jpg", ".png", ".webp", ".gif"):
        candidate = output_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def download_one(row: dict[str, str], output_dir: Path, timeout: float) -> tuple[str, str]:
    image_id = str(row["id"]).strip()
    image_url = str(row["image_url"]).strip()
    if not image_id or not image_url:
        return image_id, "missing"

    if existing := existing_image(output_dir, image_id):
        return str(existing), "exists"

    try:
        response = requests.get(
            image_url,
            timeout=timeout,
            headers={"User-Agent": "FNEWS Fakeddit research downloader/0.1"},
        )
        response.raise_for_status()
        extension = infer_extension(image_url, response.headers.get("content-type"))
        path = output_dir / f"{image_id}{extension}"
        path.write_bytes(response.content)
    except requests.RequestException:
        return image_id, "failed"

    return str(path), "downloaded"


def main() -> None:
    args = parse_args()
    tsv_path = Path(args.tsv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    required = {"id", "image_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=["id", "image_url"]).copy()
    frame["image_url"] = frame["image_url"].astype("string").str.strip()
    frame = frame[frame["image_url"].ne("")]
    if "hasImage" in frame.columns:
        frame = frame[normalize_bool_series(frame["hasImage"])]
    if args.limit is not None:
        frame = frame.head(args.limit)

    rows = frame[["id", "image_url"]].astype("string").to_dict("records")
    status_counts = {"downloaded": 0, "exists": 0, "failed": 0, "missing": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, row, output_dir, args.timeout) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="images"):
            _, status = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1

    print(status_counts)


if __name__ == "__main__":
    main()
