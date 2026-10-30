"""Bounded, resumable ReCOVery image-text acquisition for external evaluation research.

ReCOVery's ``reliability`` is retained as source-native publisher/article
credibility metadata.  It is not relabelled as Fakeddit ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 8 * 1024 * 1024
USER_AGENT = "FNEWSResearch/1.0 (bounded external evaluation pilot)"


@dataclass(frozen=True)
class DownloadResult:
    news_id: str
    source_url: str
    status: str
    image_path: str | None
    image_hash: str | None
    perceptual_hash: str | None
    bytes_downloaded: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire a bounded ReCOVery pilot.")
    parser.add_argument("--source-csv", default="data/external/recovery_pilot/recovery-news-data.csv")
    parser.add_argument("--output-dir", default="data/external/recovery_pilot")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_rows(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    available = frame.copy()
    available["image"] = available["image"].astype("string").fillna("").str.strip()
    available = available.loc[available.image.str.startswith(("http://", "https://"))].copy()
    groups = []
    for label in (0, 1):
        pool = available.loc[available.reliability.eq(label)]
        groups.append(pool.sample(n=min(limit // 2, len(pool)), random_state=seed + label))
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _save_image(content: bytes, news_id: str, target: Path) -> tuple[Path, str, str]:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("empty response or image exceeds 8 MiB")
    temporary = target / f"{news_id}.part"
    temporary.write_bytes(content)
    try:
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(
                (image.format or "").upper(), ".img"
            )
            phash = str(imagehash.phash(image.convert("RGB")))
        digest = hashlib.sha256(content).hexdigest()
        destination = target / f"{news_id}{suffix}"
        temporary.replace(destination)
        return destination, digest, phash
    except (UnidentifiedImageError, OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise


def download(row: dict[str, object], target: Path, timeout: float, retries: int) -> DownloadResult:
    news_id, source_url = str(row["news_id"]), str(row["image"])
    existing = next((path for path in target.glob(f"{news_id}.*") if path.suffix != ".part"), None)
    if existing:
        try:
            content = existing.read_bytes()
            with Image.open(existing) as image:
                image.verify()
            with Image.open(existing) as image:
                phash = str(imagehash.phash(image.convert("RGB")))
            return DownloadResult(
                news_id,
                source_url,
                "verified_reused",
                str(existing),
                hashlib.sha256(content).hexdigest(),
                phash,
                len(content),
                None,
            )
        except (UnidentifiedImageError, OSError, ValueError):
            existing.unlink(missing_ok=True)
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                source_url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"},
            )
            response.raise_for_status()
            if not response.headers.get("content-type", "").lower().startswith("image/"):
                raise ValueError("response content type is not an image")
            path, digest, phash = _save_image(response.content, news_id, target)
            return DownloadResult(news_id, source_url, "verified", str(path), digest, phash, len(response.content), None)
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1) + random.random() * 0.2)
    return DownloadResult(news_id, source_url, "failed", None, None, None, 0, error)


def duplicate_summary(manifest: pd.DataFrame) -> dict[str, int]:
    usable = manifest.download_status.isin(["verified", "verified_reused"])
    hashes = manifest.loc[usable, "image_hash"]
    exact_groups = int((hashes.value_counts() > 1).sum())
    perceptual = manifest.loc[usable, "perceptual_hash"].dropna().tolist()
    near_pairs = sum(
        int(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right) <= 2)
        for index, left in enumerate(perceptual)
        for right in perceptual[index + 1 :]
    )
    return {"exact_hash_groups": exact_groups, "near_duplicate_pairs_hamming_lte_2": near_pairs}


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free space is required for this bounded pilot.")
    source = pd.read_csv(args.source_csv)
    selected = select_rows(source, args.limit, args.seed)
    selected["news_id"] = selected["news_id"].astype(str)
    rows = selected[["news_id", "image"]].to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
        futures = [executor.submit(download, row, image_root, args.timeout, args.retries) for row in rows]
        results = [future.result() for future in as_completed(futures)]
    result_frame = pd.DataFrame([asdict(item) for item in results])
    manifest = selected.merge(result_frame, on="news_id", how="left")
    manifest = manifest.rename(
        columns={
            "image": "image_source_url",
            "reliability": "source_reliability_label",
            "status": "download_status",
        }
    )
    manifest["source_dataset"] = "ReCOVery"
    manifest["label_semantics"] = "source reliability (1=reliable, 0=unreliable); not Fakeddit truth"
    manifest["split"] = "unspecified_by_source"
    manifest.to_parquet(root / "manifest.parquet", index=False)
    manifest.to_csv(root / "manifest.csv", index=False)
    report = {
        "source_csv": str(Path(args.source_csv)),
        "selected": len(manifest),
        "verified_pairs": int(manifest.download_status.isin(["verified", "verified_reused"]).sum()),
        "failed_images": int(manifest.download_status.eq("failed").sum()),
        "bytes_verified": int(manifest.bytes_downloaded.fillna(0).sum()),
        "label_distribution_selected": {str(k): int(v) for k, v in manifest.source_reliability_label.value_counts().items()},
        "label_distribution_verified": {str(k): int(v) for k, v in manifest.loc[manifest.download_status.isin(["verified", "verified_reused"]), "source_reliability_label"].value_counts().items()},
        "duplicate_audit": duplicate_summary(manifest),
        "limit": args.limit,
        "workers": min(args.workers, 4),
        "retries": args.retries,
    }
    (root / "acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
