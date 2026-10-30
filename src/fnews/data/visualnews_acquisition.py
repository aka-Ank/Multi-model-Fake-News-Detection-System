from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from fnews.data.external_storage import external_dataset_path

ORIGIN_URL = "https://www.cs.rice.edu/~vo9/visualnews/origin.tar"
ORIGIN_BYTES = 97_916_641_280
ALLOWED_HOSTS = {"www.cs.rice.edu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run the external-drive VisualNews origin download.")
    parser.add_argument("--execute", action="store_true", help="Start the large transfer after explicit approval.")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def free_gib(path: Path) -> float:
    data = os.statvfs(path)
    return data.f_bavail * data.f_frsize / 1024**3


def plan(root: Path) -> dict[str, object]:
    destination = root / "downloads" / "visualnews" / "origin.tar"
    partial = destination.with_suffix(".tar.part")
    return {
        "url": ORIGIN_URL,
        "expected_bytes": ORIGIN_BYTES,
        "destination": str(destination),
        "partial_path": str(partial),
        "free_gib": round(free_gib(root), 2),
        "minimum_free_gib": 200,
        "archive_exists": destination.exists(),
        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
        "range_supported": True,
    }


def valid_remote(response_url: str) -> bool:
    parsed = urlparse(response_url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def download_origin(root: Path, timeout: float) -> dict[str, object]:
    state = plan(root)
    if state["free_gib"] < state["minimum_free_gib"]:
        raise RuntimeError("External volume does not meet the 200 GiB free-space requirement")
    destination = Path(str(state["destination"]))
    partial = Path(str(state["partial_path"]))
    log_path = root / "logs" / "visualnews_download_state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == ORIGIN_BYTES:
        state["status"] = "verified_existing_archive"
        log_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(ORIGIN_URL, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
        if not valid_remote(response.url):
            raise RuntimeError(f"Rejected redirect to {response.url}")
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        downloaded = offset if append else 0
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded % (512 * 1024 * 1024) < len(chunk):
                    state.update({"status": "downloading", "downloaded_bytes": downloaded, "updated_at": time.time()})
                    log_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if partial.stat().st_size != ORIGIN_BYTES:
        state.update({"status": "incomplete", "downloaded_bytes": partial.stat().st_size})
        log_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise RuntimeError("Download ended before the expected official Content-Length")
    partial.replace(destination)
    state.update({"status": "downloaded_size_verified", "downloaded_bytes": ORIGIN_BYTES, "updated_at": time.time()})
    log_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    args = parse_args()
    root = external_dataset_path()
    result = download_origin(root, args.timeout) if args.execute else plan(root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
