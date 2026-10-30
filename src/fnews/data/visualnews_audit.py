from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_COLUMNS = [
    "sample_id",
    "dataset_source",
    "original_split",
    "article_id",
    "image_id",
    "text",
    "caption",
    "image_path",
    "label",
    "label_definition",
    "article_url",
    "image_url",
    "source_publication",
    "content_hash",
    "image_hash",
    "download_status",
    "validation_status",
]


@dataclass(frozen=True)
class ArchiveInventory:
    archive: str
    bytes: int
    members_scanned: int
    json_members: int
    image_members: int
    scan_complete: bool
    sample_member: str | None
    sample_keys: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit VisualNews article archives without bulk extraction.")
    parser.add_argument("--articles-dir", default="/Users/aka_ank/Desktop/articles")
    parser.add_argument("--report-dir", default="reports/datasets")
    parser.add_argument("--output-manifest", default="data/processed/visualnews/visualnews_smoke_manifest.parquet")
    parser.add_argument("--samples-per-archive", type=int, default=10)
    parser.add_argument("--max-members-per-archive", type=int, default=5_000)
    return parser.parse_args()


def text_value(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def nested_image_fields(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        url = text_value(value, "href", "url", "image_url", "src")
        caption = text_value(value, "caption", "altText", "alt", "description")
        kind = text_value(value, "type", "primaryType", "secondaryType") or ""
        if url and ("image" in kind.lower() or any(key in value for key in ("image_url", "altText", "src"))):
            found.append({"image_url": url, "caption": caption or ""})
        for child in value.values():
            found.extend(nested_image_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_image_fields(child))
    return found


def article_record(record: dict[str, Any], archive_name: str, member_name: str) -> dict[str, object]:
    article_id = text_value(record, "id", "article_id", "uuid") or member_name
    text = text_value(record, "body", "content", "article", "text", "summary")
    title = text_value(record, "name", "title", "headline", "shortName")
    images = nested_image_fields(record)
    image = images[0] if images else {"image_url": None, "caption": None}
    combined = "\n\n".join(part for part in (title, text) if part)
    sample_id = hashlib.sha256(f"{archive_name}:{member_name}".encode()).hexdigest()[:24]
    return {
        "sample_id": sample_id,
        "dataset_source": "VisualNews article archive",
        "original_split": None,
        "article_id": article_id,
        "image_id": None,
        "text": combined or None,
        "caption": image["caption"] or None,
        "image_path": None,
        "label": None,
        "label_definition": None,
        "article_url": text_value(record, "shareUrl", "url", "article_url"),
        "image_url": image["image_url"],
        "source_publication": archive_name.removesuffix("_articles.tar.gz").removesuffix(".tar.gz"),
        "content_hash": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        "image_hash": None,
        "download_status": "not_downloaded" if image["image_url"] else "no_image_reference",
        "validation_status": "unverified_smoke_sample",
    }


def inspect_archive(
    path: Path, samples_per_archive: int, max_members: int = 5_000
) -> tuple[ArchiveInventory, list[dict[str, object]]]:
    sample_rows: list[dict[str, object]] = []
    member_count = json_count = image_count = 0
    first_member: str | None = None
    first_keys: str | None = None
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_count += 1
                if member_count > max_members:
                    break
                suffix = Path(member.name).suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                    image_count += 1
                if suffix != ".json":
                    continue
                json_count += 1
                if len(sample_rows) >= samples_per_archive:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    record = json.load(handle)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if first_member is None:
                    first_member = member.name
                    first_keys = ", ".join(sorted(record)[:30])
                sample_rows.append(article_record(record, path.name, member.name))
    except (tarfile.ReadError, EOFError, OSError) as exc:
        return ArchiveInventory(path.name, path.stat().st_size, 0, 0, 0, False, None, None, f"{type(exc).__name__}: {exc}"), []
    return (
        ArchiveInventory(
            path.name,
            path.stat().st_size,
            min(member_count, max_members),
            json_count,
            image_count,
            member_count <= max_members,
            first_member,
            first_keys,
            None,
        ),
        sample_rows,
    )


def inspect_pickle_opcodes(path: Path, maximum_opcodes: int = 20_000) -> dict[str, object]:
    keys: set[str] = set()
    unsafe_opcodes: set[str] = set()
    count = 0
    with path.open("rb") as handle:
        for opcode, argument, _ in pickletools.genops(handle):
            count += 1
            if opcode.name in {"GLOBAL", "REDUCE", "STACK_GLOBAL", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"}:
                unsafe_opcodes.add(opcode.name)
            if isinstance(argument, str) and len(argument) < 80 and argument.replace("_", "").isalnum():
                keys.add(argument)
            if count >= maximum_opcodes:
                break
    return {
        "pickle": path.name,
        "bytes": path.stat().st_size,
        "opcodes_scanned": count,
        "possible_fields": sorted(keys),
        "unsafe_deserialization_opcodes_seen": sorted(unsafe_opcodes),
        "deserialized": False,
    }


def write_reports(
    report_dir: Path,
    inventories: list[ArchiveInventory],
    pickle_inspections: list[dict[str, object]],
    samples: pd.DataFrame,
    manifest_path: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(item) for item in inventories).to_csv(report_dir / "visualnews_archive_inventory.csv", index=False)
    (report_dir / "visualnews_pickle_static_inspection.json").write_text(
        json.dumps(pickle_inspections, indent=2), encoding="utf-8"
    )
    (report_dir / "visualnews_representative_records.json").write_text(
        samples.head(20).to_json(orient="records", indent=2), encoding="utf-8"
    )
    total_json = sum(item.json_members for item in inventories)
    total_images = sum(item.image_members for item in inventories)
    with_urls = int(samples["image_url"].notna().sum()) if len(samples) else 0
    report = [
        "# VisualNews Dataset Audit",
        "",
        "## What Is Present",
        "",
        f"- Five publisher archive files; bounded scan found {total_json:,} JSON records.",
        f"- Archive-contained image files in the bounded scan: {total_images:,}.",
        "- Five Python pickle files were inspected statically only; none was deserialized.",
        f"- Smoke manifest: {len(samples):,} article records; {with_urls:,} expose an image URL; 0 verified local images.",
        "",
        "## Compatibility",
        "",
        "These are VisualNews-style article records and image references. They have no Fakeddit real/fake label and are not"
        " a supervised replacement for Fakeddit. NewsCLIPpings annotations are also absent, so matched/mismatched labels cannot"
        " be assembled yet. The smoke manifest deliberately stores `label` and `label_definition` as null.",
        "",
        "## Missing Components",
        "",
        "- VisualNews `data.json` and verified image corpus are not present.",
        "- NewsCLIPpings `data/` annotations, including train/validation/test matches, are not present.",
        "- NewsCLIPpings embeddings are not present.",
        "",
        "## Safe Next Step",
        "",
        "Acquire the official VisualNews distribution first, then the official NewsCLIPpings annotations. Estimate size and"
        " obtain approval before any multi-gigabyte download. Do not merge NewsCLIPpings falsified-pair labels with Fakeddit"
        " misinformation labels.",
        "",
        f"Smoke manifest: `{manifest_path}`",
    ]
    (report_dir / "VISUALNEWS_AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    readiness = [
        "# Dataset Readiness",
        "",
        "| Track | Current state | Ready for supervised training |",
        "|:--|:--|:--|",
        "| Fakeddit | Existing independent misinformation track | Yes, under its current sealed-test protocol |",
        "| VisualNews articles | JSON archives and unverified image URLs | No |",
        "| NewsCLIPpings | Not present | No |",
        "",
        "VisualNews records may be used later for image-caption representation research after verified images are acquired."
        " They must remain a separate provenance-tagged track.",
    ]
    (report_dir / "DATASET_READINESS.md").write_text("\n".join(readiness), encoding="utf-8")
    plan = [
        "# Integration Plan",
        "",
        "1. Retain Fakeddit as the only real/fake supervised classification track.",
        "2. Acquire official VisualNews `data.json` and images after a storage estimate and explicit approval.",
        "3. Acquire official NewsCLIPpings annotations; use `falsified` only for image-caption mismatch detection.",
        "4. Build separate manifests using the standardized schema and enforce article/image split isolation.",
        "5. Begin with frozen CLIP image/text embeddings and a lightweight alignment classifier.",
        "6. Compare any representation pretraining transfer against Fakeddit without mixing labels or test splits.",
    ]
    (report_dir / "INTEGRATION_PLAN.md").write_text("\n".join(plan), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    articles_dir = Path(args.articles_dir)
    archives = sorted(articles_dir.glob("*.tar.gz"))
    inventories: list[ArchiveInventory] = []
    rows: list[dict[str, object]] = []
    for archive in archives:
        inventory, samples = inspect_archive(archive, args.samples_per_archive, args.max_members_per_archive)
        inventories.append(inventory)
        rows.extend(samples)
    samples = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(manifest_path, index=False)
    pickle_inspections = [inspect_pickle_opcodes(path) for path in sorted(articles_dir.glob("*.p"))]
    write_reports(Path(args.report_dir), inventories, pickle_inspections, samples, manifest_path)
    return {"archives": len(archives), "smoke_records": len(samples), "manifest": str(manifest_path)}


def main() -> None:
    print(json.dumps(run_audit(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
