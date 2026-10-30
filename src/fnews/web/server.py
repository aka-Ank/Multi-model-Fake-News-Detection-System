from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

from fnews.web.extract import FetchError, extract_article, fetch_bytes, public_url
from fnews.web.models import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_TEXT, ModelRegistry
from fnews.web.store import Store

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    Path(os.environ.get("FNEWS_APP_DATA", ROOT / "data" / "research_app")).expanduser().resolve()
)
HOST = os.environ.get("FNEWS_HOST", "127.0.0.1")
PORT = int(os.environ.get("FNEWS_PORT", "8765"))
MAX_REQUEST_BYTES = 12 * 1024 * 1024
ALLOWED_ORIGINS = {"http://127.0.0.1:8765", "http://localhost:8765"}
logging.basicConfig(
    level=os.environ.get("FNEWS_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("fnews.web")
store = Store(DATA_DIR)
registry = ModelRegistry()
workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fnews-job")
analysis_lock = threading.Lock()


def _validate_image(data: bytes) -> tuple[int, int, str]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be between 1 byte and 8 MB.")
    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            fmt = (image.format or "").upper()
            if fmt not in {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}:
                raise ValueError("This image format is not supported.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the 20 megapixel limit.")
            image.verify()
        with Image.open(__import__("io").BytesIO(data)) as image:
            image.load()
            return image.width, image.height, fmt.lower()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The selected file is not a valid supported image.") from exc


def _persist_image(data: bytes, key: str | None = None) -> str:
    _validate_image(data)
    key = key or f"{uuid.uuid4().hex}.img"
    if not re.fullmatch(r"[a-f0-9]{32}\.img", key):
        raise ValueError("Invalid generated image key.")
    target = store.image_root / key
    target.write_bytes(data)
    return key


def _run_extract(job_id: str, url: str) -> None:
    store.set_job(job_id, status="running")
    try:
        article = extract_article(url)
        article_id = store.save_article(article)
        store.set_job(job_id, status="completed", result={
            "article_id": article_id, "mode": "review_only", "primary_image_index": None,
            "image_key": None, "automatic_analysis": "disabled_manual_confirmation_required",
        })
    except Exception as exc:
        logger.info("job_failed id=%s kind=article_extract type=%s", job_id, type(exc).__name__)
        store.set_job(job_id, status="failed", error=str(exc)[:500])


def _run_analysis(
    job_id: str,
    analysis_id: str,
    text: str,
    image_key: str | None,
    image_source_url: str | None,
    model_ids: list[str],
) -> None:
    store.set_job(job_id, status="running")
    store.set_analysis_status(analysis_id, "running")
    image_path = store.image_root / image_key if image_key else None
    try:
        with analysis_lock:
            predictions = registry.predict(text, image_path, model_ids)
        store.finish_analysis(analysis_id, predictions)
        result = store.get_analysis(analysis_id)
        store.set_job(
            job_id,
            status="completed" if result and result["status"] != "failed" else "failed",
            result={"analysis_id": analysis_id},
        )
    except Exception as exc:
        logger.info("job_failed id=%s kind=inference type=%s", job_id, type(exc).__name__)
        store.finish_analysis(
            analysis_id,
            [
                {
                    "model_id": "analysis",
                    "status": "failed",
                    "error": "Inference could not be completed.",
                }
            ],
        )
        store.set_job(job_id, status="failed", error="Inference could not be completed.")


class Handler(BaseHTTPRequestHandler):
    server_version = "FNEWSResearch/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("http client=%s message=%s", self.client_address[0], fmt % args)

    def _send(
        self, status: int, payload: object, *, content_type: str = "application/json; charset=utf-8"
    ) -> None:
        if isinstance(payload, bytes):
            body = payload
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_id = self.headers.get("X-Request-ID", str(uuid.uuid4()))[:100]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", request_id)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' data:; img-src 'self' data: https:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > MAX_REQUEST_BYTES:
            raise ValueError("Request body is empty or exceeds the 12 MB limit.")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ValueError("Content-Type must be application/json.")
        try:
            parsed = json.loads(self.rfile.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        if origin not in ALLOWED_ORIGINS:
            self._send(403, {"error": "Origin is not allowed."})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/" or path == "/index.html":
            self._send(
                200,
                (ROOT / "src/fnews/web/index.html").read_text(encoding="utf-8"),
                content_type="text/html; charset=utf-8",
            )
        elif path == "/api/health/live":
            self._send(200, {"status": "live"})
        elif path == "/api/health/ready":
            self._send(200, {"status": "ready", "database": store.path.exists()})
        elif path == "/api/models":
            self._send(200, {"models": registry.models()})
        elif path == "/api/analyses":
            query = parse_qs(urlsplit(self.path).query)
            self._send(
                200,
                store.list_analyses(
                    _int_query(query, "page", 1), _int_query(query, "page_size", 20)
                ),
            )
        elif path.startswith("/api/jobs/"):
            item = store.get_job(path.rsplit("/", 1)[-1])
            self._send(200 if item else 404, item or {"error": "Job not found."})
        elif path.startswith("/api/articles/"):
            item = store.get_article(path.rsplit("/", 1)[-1])
            self._send(200 if item else 404, item or {"error": "Article not found."})
        elif path.startswith("/api/analyses/"):
            item = store.get_analysis(path.rsplit("/", 1)[-1])
            self._send(200 if item else 404, item or {"error": "Analysis not found."})
        elif path.startswith("/media/"):
            key = path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[a-f0-9]{32}\.img", key):
                self._send(404, {"error": "Image not found."})
                return
            target = store.image_root / key
            if not target.is_file():
                self._send(404, {"error": "Image not found."})
                return
            data = target.read_bytes()
            try:
                _, _, fmt = _validate_image(data)
                mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                    "gif": "image/gif",
                    "bmp": "image/bmp",
                    "tiff": "image/tiff",
                }.get(fmt, "application/octet-stream")
                self._send(200, data, content_type=mime)
            except ValueError:
                self._send(404, {"error": "Image not found."})
        else:
            self._send(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._read_json()
            match = re.fullmatch(r"/api/articles/([a-f0-9-]{36})/images/(\d+)/preview", path)
            if match:
                article = store.get_article(match.group(1))
                index = int(match.group(2))
                if not article or index >= len(article["images"]):
                    self._send(404, {"error": "Candidate image not found."})
                    return
                candidate = article["images"][index]
                data, _, _ = fetch_bytes(candidate["source_url"], image=True)
                key = _persist_image(data)
                self._send(200, {"image_key": key, "preview_url": f"/media/{key}"})
                return
            if path == "/api/articles/extract":
                url = str(body.get("url", ""))
                public_url(url)
                job_id = store.create_job("article_extract")
                workers.submit(_run_extract, job_id, url)
                self._send(202, {"job_id": job_id, "status": "queued"})
                return
            if path == "/api/analyses":
                text = str(body.get("text", ""))
                model_ids = body.get("model_ids") or ["fakeddit-tfidf-linearsvc-research-v1"]
                if not isinstance(model_ids, list) or any(
                    not isinstance(x, str) for x in model_ids
                ):
                    raise ValueError("model_ids must be a list of model identifiers.")
                allowed = {
                    item["id"] for item in registry.models() if item["availability"] == "available"
                }
                unknown = set(model_ids) - allowed
                if unknown:
                    raise ValueError("One or more selected models are unavailable or unknown.")
                text_required = any(
                    model_id
                    in {"fakeddit-tfidf-linearsvc-research-v1", "fakeddit-text-embedding-50k-seed42-v1"}
                    for model_id in model_ids
                )
                image_required = any(
                    model_id
                    in {"fakeddit-image-cnn-pilot-seed42-v1"}
                    for model_id in model_ids
                )
                # The simplified dashboard permits a validated image to be
                # retained as visible context beside text-only inference. It
                # is not supplied to a text model or represented as fusion.
                image_accepted = image_required or "fakeddit-weighted-late-fusion-v1" in model_ids or model_ids == [
                    "fakeddit-tfidf-linearsvc-research-v1"
                ]
                if (text_required and not text.strip()) or len(text) > MAX_TEXT:
                    raise ValueError(f"Selected text analysis requires 1 to {MAX_TEXT} characters.")
                image_key = None
                image_url = str(body.get("image_url", "")).strip() or None
                image_data = body.get("image_base64")
                image_sha256 = None
                image_width = image_height = None
                image_format = None
                if body.get("article_image_index") is not None:
                    article_id = str(body.get("article_id", "")).strip()
                    article = store.get_article(article_id) if article_id else None
                    try:
                        image_index = int(body["article_image_index"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Selected article image index is invalid.") from exc
                    if not article or image_index < 0 or image_index >= len(article["images"]):
                        raise ValueError("Selected article image was not found.")
                    image_key = str(body.get("image_key", ""))
                    if (
                        not re.fullmatch(r"[a-f0-9]{32}\.img", image_key)
                        or not (store.image_root / image_key).is_file()
                    ):
                        raise ValueError(
                            "Preview the selected image before submitting the analysis."
                        )
                    image_url = article["images"][image_index]["source_url"]
                    image_bytes = (store.image_root / image_key).read_bytes()
                    image_width, image_height, image_format = _validate_image(image_bytes)
                    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                elif image_data:
                    try:
                        binary = base64.b64decode(str(image_data), validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise ValueError("Image data is not valid base64.") from exc
                    image_width, image_height, image_format = _validate_image(binary)
                    image_key = _persist_image(binary)
                    image_sha256 = hashlib.sha256(binary).hexdigest()
                elif image_url:
                    raise ValueError(
                        "Use a selected article image preview or upload an image file."
                    )
                if image_key and not image_accepted:
                    raise ValueError("An image was supplied but no selected model uses it.")
                if image_required and not image_key:
                    raise ValueError("The selected model requires an image.")
                article_id = str(body.get("article_id", "")).strip() or None
                if article_id and not store.get_article(article_id):
                    raise ValueError("Article reference was not found.")
                if article_id and image_key:
                    input_type = "article_pair"
                elif article_id:
                    input_type = "article_text"
                elif image_key and text.strip():
                    input_type = "text_image"
                elif image_key:
                    input_type = "image"
                else:
                    input_type = "text"
                analysis_id = store.create_analysis(
                    input_type,
                    text,
                    image_key,
                    image_url,
                    image_sha256,
                    image_width,
                    image_height,
                    image_format,
                    article_id,
                )
                job_id = store.create_job("inference")
                workers.submit(
                    _run_analysis, job_id, analysis_id, text, image_key, image_url, model_ids
                )
                self._send(202, {"analysis_id": analysis_id, "job_id": job_id, "status": "queued"})
                return
            self._send(404, {"error": "Not found."})
        except (ValueError, FetchError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception(
                "request_failed request_id=%s type=%s",
                self.headers.get("X-Request-ID", "-"),
                type(exc).__name__,
            )
            self._send(500, {"error": "The request could not be completed."})


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [str(default)])[0])
    except ValueError:
        return default


def main() -> None:
    if HOST not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            "FNEWS_HOST must be loopback; public binding is not supported by this local research app."
        )
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.daemon_threads = True
    logger.info("service_started host=%s port=%s database=%s", HOST, PORT, store.path)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        workers.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
