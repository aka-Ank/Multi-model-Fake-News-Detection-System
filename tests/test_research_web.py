from __future__ import annotations

import socket
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from fnews.web import extract, server
from fnews.web.extract import FetchError, public_url, resolve_public
from fnews.web.models import ModelRegistry
from fnews.web.store import Store


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd", "http://user:pass@example.com/story", "http://127.0.0.1/story",
        "http://10.0.0.8/story", "http://169.254.169.254/latest/meta-data/", "http://[::1]/story",
        "http://[fc00::1]/story", "http://localhost/story", "https://example.com:8080/story",
    ],
)
def test_rejects_unsafe_article_urls(url: str) -> None:
    with pytest.raises(FetchError):
        public_url(url)


def test_rejects_hostname_that_resolves_to_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.4", 80))
    ])
    with pytest.raises(FetchError, match="private or reserved"):
        resolve_public("news.example", 80)


def test_article_extraction_returns_plain_text_and_bounded_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b'''<html><head><title>Fallback title</title><meta property="og:title" content="A verified headline">
    <meta property="og:site_name" content="Example Gazette"></head><body><article>
    <p>This is a sufficiently long article paragraph with details about a public research announcement.</p>
    <figure><img src="/media/story.jpg" alt="Research team"><figcaption>Researchers in the lab</figcaption></figure>
    <script>secret()</script></article></body></html>'''
    monkeypatch.setattr(extract, "fetch_bytes", lambda url, image=False: (html, "text/html", "https://news.example/story"))
    result = extract.extract_article("https://news.example/story")
    assert result["headline"] == "A verified headline"
    assert "public research announcement" in result["body"]
    assert result["publisher"] == "Example Gazette"
    assert result["images"][0]["source_url"] == "https://news.example/media/story.jpg"
    assert result["images"][0]["caption"] == "Researchers in the lab"
    assert result["primary_image_index"] == 0
    assert "secret" not in result["body"]


def test_primary_image_ranking_prefers_structured_article_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b'''<html><head><meta property="og:image" content="/logo.png"><script type="application/ld+json">
    {"@type":"NewsArticle","image":"https://news.example/images/lead.jpg"}</script></head><body><article>
    <p>This paragraph is long enough to form the article body for this controlled extraction fixture.</p>
    <img src="/images/inline.jpg" alt="Inline report image"></article></body></html>'''
    monkeypatch.setattr(extract, "fetch_bytes", lambda url, image=False: (html, "text/html", "https://news.example/story"))
    result = extract.extract_article("https://news.example/story")
    assert result["images"][0]["source_url"] == "https://news.example/images/lead.jpg"
    assert result["images"][0]["source_kind"] == "structured_data"
    assert result["primary_image_index"] == 0


def test_low_confidence_body_image_requires_manual_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b'''<html><body><article><p>This paragraph is long enough to form the article body for this controlled extraction fixture.</p>
    <img src="/images/inline.jpg" alt="photo"></article></body></html>'''
    monkeypatch.setattr(extract, "fetch_bytes", lambda url, image=False: (html, "text/html", "https://news.example/story"))
    result = extract.extract_article("https://news.example/story")
    assert result["images"][0]["ranking_score"] < 70
    assert result["primary_image_index"] is None
    assert any("confidence is low" in warning for warning in result["warnings"])


def test_live_blog_uses_only_a_coherent_update_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b'''<html><head><meta property="og:title" content="War live blog | Example"><meta property="og:image" content="/global.jpg"></head>
    <body><main><section><time>2026-10-27T12:00:00Z</time><h2>Latest verified update heading</h2>
    <p>This sufficiently detailed update belongs with its own image and must not be paired with the global lead image.</p>
    <img src="/update.jpg" alt="Update photograph"></section></main></body></html>'''
    monkeypatch.setattr(extract, "fetch_bytes", lambda url, image=False: (html, "text/html", "https://news.example/liveblog/story"))
    result = extract.extract_article("https://news.example/liveblog/story")
    assert result["is_live_blog"] is True
    assert result["page_title"] == "War live blog | Example"
    assert result["headline"] == "Latest verified update heading"
    assert result["images"][0]["source_url"] == "https://news.example/update.jpg"
    assert result["primary_image_index"] == 0
    assert any("continuously updated" in warning for warning in result["warnings"])


class _FixtureRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path | None, list[str]]] = []

    def predict(self, text: str, image_path: Path | None, model_ids: list[str]) -> list[dict[str, object]]:
        self.calls.append((text, image_path, model_ids))
        return [{"model_id": model_ids[0], "status": "completed", "task": "fake_news_classification", "raw_score_fake": 0.2}]


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (640, 400), (80, 110, 150)).save(output, format="JPEG")
    return output.getvalue()


def test_url_workflow_requires_manual_image_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_store = Store(tmp_path)
    fixture_registry = _FixtureRegistry()
    article = {
        "submitted_url": "https://news.example/story", "canonical_url": "https://news.example/story",
        "headline": "Fixture headline", "body": "Fixture body", "warnings": [],
        "images": [{"source_url": "https://cdn.example/lead.jpg", "ranking_score": 100}],
        "primary_image_index": 0, "content_hash": "fixture",
    }
    monkeypatch.setattr(server, "store", fixture_store)
    monkeypatch.setattr(server, "registry", fixture_registry)
    monkeypatch.setattr(server, "extract_article", lambda url: article)
    monkeypatch.setattr(server, "fetch_bytes", lambda url, image=False: (_image_bytes(), "image/jpeg", url))
    job_id = fixture_store.create_job("article_extract")
    server._run_extract(job_id, "https://news.example/story")
    job = fixture_store.get_job(job_id)
    assert job and job["status"] == "completed"
    assert job["result"]["mode"] == "review_only"
    assert job["result"]["automatic_analysis"] == "disabled_manual_confirmation_required"
    assert fixture_registry.calls == []


def test_url_workflow_never_fetches_image_before_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_store = Store(tmp_path)
    fixture_registry = _FixtureRegistry()
    article = {
        "submitted_url": "https://news.example/story", "canonical_url": "https://news.example/story",
        "headline": "Fixture headline", "body": "Fixture body", "warnings": [],
        "images": [{"source_url": "https://cdn.example/broken.jpg", "ranking_score": 100}],
        "primary_image_index": 0, "content_hash": "fixture",
    }
    monkeypatch.setattr(server, "store", fixture_store)
    monkeypatch.setattr(server, "registry", fixture_registry)
    monkeypatch.setattr(server, "extract_article", lambda url: article)
    monkeypatch.setattr(server, "fetch_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(FetchError("broken")))
    job_id = fixture_store.create_job("article_extract")
    server._run_extract(job_id, "https://news.example/story")
    job = fixture_store.get_job(job_id)
    assert job and job["result"]["mode"] == "review_only"
    assert job["result"]["primary_image_index"] is None
    assert fixture_registry.calls == []


def test_store_persists_job_article_and_analysis(tmp_path: Path) -> None:
    store = Store(tmp_path)
    job_id = store.create_job("fixture")
    store.set_job(job_id, status="running")
    store.set_job(job_id, status="completed", result={"ok": True})
    article_id = store.save_article({
        "submitted_url": "https://news.example/a", "content_hash": "abc", "body": "story", "images": [], "warnings": []
    })
    analysis_id = store.create_analysis("text", "exact title", None, None, None, None, None, None, article_id)
    store.finish_analysis(analysis_id, [{"status": "completed", "task": "fake_news_classification"}])
    reopened = Store(tmp_path)
    assert reopened.get_job(job_id)["result"] == {"ok": True}
    assert reopened.get_article(article_id)["body"] == "story"
    analysis = reopened.get_analysis(analysis_id)
    assert analysis["exact_text"] == "exact title"
    assert analysis["status"] == "completed"


def test_saved_checkpoints_are_loaded_for_real_inference(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (80, 60), (90, 130, 180)).save(image_path)
    registry = ModelRegistry()
    text = registry.predict(
        "Researchers report a new public health study and publish their findings.", None,
        ["fakeddit-tfidf-linearsvc-research-v1"],
    )
    legacy_text = registry.predict(
        "Researchers report a new public health study and publish their findings.", None,
        ["fakeddit-text-embedding-50k-seed42-v1"],
    )
    image = registry.predict("", image_path, ["fakeddit-image-cnn-pilot-seed42-v1"])
    assert text[0]["status"] == "completed"
    assert text[0]["score_name"] == "LinearSVC non-true-positive decision margin"
    assert legacy_text[0]["status"] == "completed"
    assert image[0]["status"] == "completed"
    assert 0 <= image[0]["raw_score_fake"] <= 1
    models = {item["id"]: item for item in registry.models()}
    assert models["fakeddit-clip-gated-fusion-v1"]["availability"] == "archived_inactive"
    assert models["image-text-mismatch-unavailable"]["availability"] == "unavailable"
