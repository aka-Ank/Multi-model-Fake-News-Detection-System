from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SUCCESS = "success"


@dataclass(frozen=True)
class DownloadResult:
    image_id: str
    image_url: str
    label: int
    status: str
    path: str | None
    sha256: str | None
    bytes_downloaded: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable, training-only Fakeddit image acquisition.")
    parser.add_argument("--manifest", default="data/processed/fakeddit/image_manifest.parquet")
    parser.add_argument("--output-dir", default="data/raw/fakeddit/images")
    parser.add_argument("--state-db", default="reports/image_research/acquisition/training_images.sqlite")
    parser.add_argument("--report-dir", default="reports/image_research/acquisition")
    parser.add_argument("--attempt-limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def free_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS acquisition (
            image_id TEXT PRIMARY KEY,
            image_url TEXT NOT NULL,
            label INTEGER NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            local_path TEXT,
            sha256 TEXT,
            bytes_downloaded INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    return connection


def already_completed(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT image_id FROM acquisition WHERE status = ?", (SUCCESS,)).fetchall()
    return {str(row[0]) for row in rows}


def previous_attempts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT image_id, attempts FROM acquisition").fetchall()
    return {str(image_id): int(attempts) for image_id, attempts in rows}


def select_training_candidates(
    frame: pd.DataFrame,
    completed: set[str],
    attempts: dict[str, int],
    limit: int,
    max_retries: int,
    seed: int,
) -> pd.DataFrame:
    candidates = frame.copy()
    urls = candidates["image_url"].astype("string").fillna("").str.strip()
    candidates = candidates.loc[
        candidates["split"].eq("train")
        & urls.ne("")
        & ~candidates["downloaded"].fillna(False)
        & ~candidates["id"].astype(str).isin(completed)
    ].copy()
    candidates["attempts"] = candidates["id"].astype(str).map(attempts).fillna(0).astype(int)
    candidates = candidates.loc[candidates["attempts"] < max_retries]
    selected: list[pd.DataFrame] = []
    per_label = limit // 2
    for label in (0, 1):
        pool = candidates.loc[candidates["2_way_label"].eq(label)]
        selected.append(pool.sample(n=min(per_label, len(pool)), random_state=seed + label))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < limit:
        remainder = candidates.loc[~candidates["id"].isin(result["id"])]
        result = pd.concat(
            [result, remainder.sample(n=min(limit - len(result), len(remainder)), random_state=seed + 99)],
            ignore_index=True,
        )
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def image_suffix(image: Image.Image, url: str) -> str:
    formats = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}
    if image.format in formats:
        return formats[image.format]
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"} else ".jpg"


def validate_and_store(content: bytes, image_id: str, image_url: str, output_dir: Path) -> tuple[Path, str]:
    if not content:
        raise ValueError("empty response body")
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            suffix = image_suffix(image, image_url)
        digest = hashlib.sha256(content).hexdigest()
        destination = output_dir / f"{image_id}{suffix}"
        temporary.replace(destination)
        return destination, digest
    except (UnidentifiedImageError, OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise


def download_one(row: dict[str, object], output_dir: Path, timeout: float, max_retries: int) -> DownloadResult:
    image_id = str(row["id"])
    image_url = str(row["image_url"])
    label = int(row["2_way_label"])
    error: str | None = None
    for retry in range(max_retries):
        try:
            response = requests.get(
                image_url,
                timeout=timeout,
                headers={"User-Agent": "FNEWS research image recovery/1.0"},
            )
            response.raise_for_status()
            path, digest = validate_and_store(response.content, image_id, image_url, output_dir)
            return DownloadResult(image_id, image_url, label, SUCCESS, str(path), digest, len(response.content), None)
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            if retry + 1 < max_retries:
                time.sleep(min(8.0, 0.5 * 2**retry) + random.uniform(0.0, 0.2))
    return DownloadResult(image_id, image_url, label, "failed", None, None, 0, error)


def record_result(connection: sqlite3.Connection, result: DownloadResult) -> None:
    connection.execute(
        """
        INSERT INTO acquisition (image_id, image_url, label, status, attempts, local_path, sha256, bytes_downloaded, error)
        VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            status=excluded.status, attempts=acquisition.attempts + 1, local_path=excluded.local_path,
            sha256=excluded.sha256, bytes_downloaded=excluded.bytes_downloaded, error=excluded.error,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            result.image_id,
            result.image_url,
            result.label,
            result.status,
            result.path,
            result.sha256,
            result.bytes_downloaded,
            result.error,
        ),
    )


def run_acquisition(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if free_gib(output_dir) < args.min_free_gib:
        raise RuntimeError(f"Insufficient free disk space under {output_dir}; requested minimum is {args.min_free_gib} GiB")
    frame = pd.read_parquet(args.manifest)
    connection = initialize_database(Path(args.state_db))
    try:
        candidates = select_training_candidates(
            frame, already_completed(connection), previous_attempts(connection), args.attempt_limit, args.max_retries, args.seed
        )
        results: list[DownloadResult] = []
        rows = candidates[["id", "image_url", "2_way_label"]].to_dict("records")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, row, output_dir, args.timeout, args.max_retries) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="training images"):
                result = future.result()
                record_result(connection, result)
                connection.commit()
                results.append(result)
    finally:
        connection.close()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame([asdict(result) for result in results])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_frame.to_csv(report_dir / f"batch_{timestamp}.csv", index=False)
    summary = {
        "attempted": len(results),
        "downloaded": int((result_frame["status"] == SUCCESS).sum()) if len(result_frame) else 0,
        "failed": int((result_frame["status"] == "failed").sum()) if len(result_frame) else 0,
        "labels_attempted": result_frame["label"].value_counts().sort_index().to_dict() if len(result_frame) else {},
        "labels_downloaded": result_frame.loc[result_frame["status"].eq(SUCCESS), "label"].value_counts().sort_index().to_dict() if len(result_frame) else {},
        "free_gib_after": round(free_gib(output_dir), 2),
        "batch_csv": str(report_dir / f"batch_{timestamp}.csv"),
    }
    (report_dir / "latest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_acquisition(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
