import io
import json
import tarfile

from fnews.data.visualnews_audit import article_record, inspect_archive


def test_article_record_keeps_label_null_and_extracts_nested_image() -> None:
    record = {
        "id": "article-1",
        "name": "Headline",
        "body": "Article body",
        "relations": [{"content": {"type": "news.image", "href": "https://example.com/image.jpg", "altText": "Caption"}}],
    }

    result = article_record(record, "bbc_1.tar.gz", "record.json")

    assert result["label"] is None
    assert result["image_url"] == "https://example.com/image.jpg"
    assert result["caption"] == "Caption"


def test_archive_inspection_streams_json_samples(tmp_path) -> None:
    path = tmp_path / "sample.tar.gz"
    payload = json.dumps({"id": "a", "name": "title"}).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("json/a.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    inventory, rows = inspect_archive(path, samples_per_archive=1)

    assert inventory.json_members == 1
    assert inventory.image_members == 0
    assert rows[0]["article_id"] == "a"


def test_archive_inspection_records_invalid_gzip(tmp_path) -> None:
    path = tmp_path / "invalid.tar.gz"
    path.write_bytes(b"json,not a gzip archive")

    inventory, rows = inspect_archive(path, samples_per_archive=1)

    assert inventory.error is not None
    assert rows == []
