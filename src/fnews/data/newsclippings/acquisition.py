from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_URL = "https://huggingface.co/g-luo/news-clippings/resolve/main"
SPLITS = ("merged_balanced", "person_sbert_text_text", "scene_resnet_place", "semantics_clip_text_image", "semantics_clip_text_text")
PARTITIONS = ("train", "val", "test")
ALLOWED_HOSTS = {"huggingface.co", "cdn-lfs.hf.co", "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"}


@dataclass(frozen=True)
class DownloadRecord:
    relative_path: str
    url: str
    status: str
    bytes_downloaded: int
    sha256: str | None
    error: str | None


def annotation_specs() -> list[tuple[str, str]]:
    return [
        (f"data/{dataset_split}/{partition}.json", f"{BASE_URL}/data/{dataset_split}/{partition}.json")
        for dataset_split in SPLITS
        for partition in PARTITIONS
    ]


def safe_remote_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, timeout: float = 30.0) -> DownloadRecord:
    if not safe_remote_url(url):
        return DownloadRecord(str(destination), url, "rejected_url", 0, None, "URL is not an approved HTTPS host")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return DownloadRecord(str(destination), url, "exists", destination.stat().st_size, sha256_file(destination), None)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True, headers=headers) as response:
            if not safe_remote_url(response.url):
                return DownloadRecord(str(destination), url, "rejected_redirect", 0, None, response.url)
            response.raise_for_status()
            append = existing > 0 and response.status_code == 206
            if not append:
                existing = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(destination)
        return DownloadRecord(str(destination), url, "downloaded", destination.stat().st_size, sha256_file(destination), None)
    except requests.RequestException as exc:
        return DownloadRecord(str(destination), url, "failed", existing, None, f"{type(exc).__name__}: {exc}")


def acquire_annotations(root: Path, log_path: Path) -> list[DownloadRecord]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = [download_file(url, root / relative_path) for relative_path, url in annotation_specs()]
    with log_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def free_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3
