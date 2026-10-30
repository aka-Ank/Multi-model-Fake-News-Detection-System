from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import threading
import time
from html import unescape
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 4
TIMEOUT_SECONDS = 8
REQUEST_DEADLINE_SECONDS = 20
USER_AGENT = "FNEWSResearchBot/1.0 (+local research; respects robots.txt)"
_host_lock = threading.Lock()
_host_last_request: dict[str, float] = {}
_robots_cache: dict[str, tuple[float, RobotFileParser | None]] = {}
_fetch_lock = threading.Lock()


class FetchError(ValueError):
    pass


def _live_segment(body_node: object) -> dict[str, object] | None:
    """Return one self-contained update only when its text and image co-occur."""
    if not isinstance(body_node, BeautifulSoup) and not getattr(body_node, "select", None):
        return None
    candidates = []
    for node in body_node.select("article, section"):
        heading_node = node.find(["h1", "h2", "h3"])
        paragraphs = [" ".join(p.get_text(" ", strip=True).split()) for p in node.find_all("p")]
        text = "\n\n".join(part for part in paragraphs if len(part) >= 30)
        image = node.find("img")
        source_url = (image.get("src") or image.get("data-src")) if image else None
        if not heading_node or len(text) < 80 or not source_url:
            continue
        heading = " ".join(heading_node.get_text(" ", strip=True).split())[:1000]
        if not heading:
            continue
        candidates.append(
            {
                "headline": heading,
                "body": text[:100_000],
                "image_url": str(source_url),
                "has_timestamp": bool(node.find("time")),
            }
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["has_timestamp"], len(str(item["body"]))), reverse=True)[0]


def public_url(url: str) -> tuple[str, str, int, str]:
    try:
        parsed = urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise FetchError("The article URL is malformed.") from exc
    if scheme not in {"http", "https"}:
        raise FetchError("Only public HTTP and HTTPS article URLs are accepted.")
    if not host or parsed.username is not None or parsed.password is not None:
        raise FetchError("URL credentials are not allowed.")
    if port not in {80, 443}:
        raise FetchError("Only standard HTTP and HTTPS ports are accepted.")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise FetchError("Private or local hostnames are not allowed.")
    try:
        direct_ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        direct_ip = None
    if direct_ip and not direct_ip.is_global:
        raise FetchError("Private or reserved network addresses are not allowed.")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return scheme, host, port, path


def resolve_public(host: str, port: int) -> list[tuple[int, str]]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise FetchError("The hostname could not be resolved.") from exc
    addresses: list[tuple[int, str]] = []
    for family, _, _, _, sockaddr in records:
        address = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        if not address.is_global:
            raise FetchError("The hostname resolves to a private or reserved address.")
        item = (family, sockaddr[0])
        if item not in addresses:
            addresses.append(item)
    if not addresses:
        raise FetchError("The hostname has no usable public address.")
    return addresses


def _open_response(url: str, max_bytes: int) -> tuple[int, dict[str, str], bytes, str]:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        scheme, host, port, path = public_url(current)
        addresses = resolve_public(host, port)
        family, address = addresses[0]
        with _host_lock:
            delay = 1.0 - (time.monotonic() - _host_last_request.get(host, 0.0))
            if delay > 0:
                time.sleep(delay)
            _host_last_request[host] = time.monotonic()
        try:
            raw = socket.socket(family, socket.SOCK_STREAM)
            raw.settimeout(TIMEOUT_SECONDS)
            target = (address, port) if family == socket.AF_INET else (address, port, 0, 0)
            raw.connect(target)
            if scheme == "https":
                raw = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
            connection = http.client.HTTPConnection(host, port, timeout=TIMEOUT_SECONDS)
            connection.sock = raw
            connection.putrequest("GET", path, skip_accept_encoding=True)
            shown_host = f"[{host}]" if ":" in host else host
            default_port = 443 if scheme == "https" else 80
            connection.putheader("Host", shown_host if port == default_port else f"{shown_host}:{port}")
            connection.putheader("User-Agent", USER_AGENT)
            connection.putheader("Accept", "text/html,application/xhtml+xml,image/*;q=0.8,*/*;q=0.1")
            connection.putheader("Accept-Encoding", "identity")
            connection.putheader("Connection", "close")
            connection.endheaders()
            response = connection.getresponse()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise FetchError("The source could not be reached safely or timed out.") from exc
        headers = {key.lower(): value for key, value in response.getheaders()}
        if response.status in {301, 302, 303, 307, 308}:
            location = headers.get("location")
            connection.close()
            if not location or redirect_count >= MAX_REDIRECTS:
                raise FetchError("The source redirected too many times or returned an invalid redirect.")
            current = urljoin(current, location)
            public_url(current)  # Validate before the next request.
            continue
        if response.status in {401, 403, 402}:
            connection.close()
            raise FetchError("The source denied access; login and paywall bypass are not supported.")
        if response.status < 200 or response.status >= 300:
            connection.close()
            raise FetchError(f"The source returned HTTP {response.status}.")
        if int(headers.get("content-length", "0") or 0) > max_bytes:
            connection.close()
            raise FetchError("The source response exceeds the size limit.")
        blocks, count = [], 0
        deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FetchError("The source exceeded the total request time limit.")
                if connection.sock is not None:
                    connection.sock.settimeout(min(TIMEOUT_SECONDS, remaining))
                try:
                    piece = response.read(min(65536, max_bytes - count + 1))
                except (OSError, http.client.HTTPException) as exc:
                    raise FetchError("The source timed out while streaming its response.") from exc
                if not piece:
                    break
                count += len(piece)
                if count > max_bytes:
                    raise FetchError("The source response exceeds the size limit.")
                blocks.append(piece)
        finally:
            connection.close()
        return response.status, headers, b"".join(blocks), current
    raise FetchError("The source redirected too many times.")


def _robots_allowed(url: str) -> bool:
    scheme, host, port, _ = public_url(url)
    origin = f"{scheme}://{host}" + (f":{port}" if port != (443 if scheme == "https" else 80) else "")
    with _fetch_lock:
        cached = _robots_cache.get(origin)
    if cached and time.monotonic() - cached[0] < 600:
        parser = cached[1]
    else:
        robots_url = origin + "/robots.txt"
        try:
            _, headers, data, _ = _open_response(robots_url, 256 * 1024)
            if "text/plain" not in headers.get("content-type", "text/plain"):
                parser = None
            else:
                parser = RobotFileParser(robots_url)
                parser.parse(data.decode("utf-8", errors="replace").splitlines())
        except FetchError as exc:
            if "denied access" in str(exc).lower():
                parser = RobotFileParser(robots_url)
                parser.parse(["User-agent: *", "Disallow: /"])
            else:
                parser = None
        with _fetch_lock:
            _robots_cache[origin] = (time.monotonic(), parser)
    return parser is None or parser.can_fetch(USER_AGENT, url)


def fetch_bytes(url: str, *, image: bool = False) -> tuple[bytes, str, str]:
    if not _robots_allowed(url):
        raise FetchError("robots.txt disallows access to this source.")
    _, headers, content, final_url = _open_response(url, MAX_IMAGE_BYTES if image else MAX_HTML_BYTES)
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if image:
        if not content_type.startswith("image/"):
            raise FetchError("The selected source did not return an image.")
    elif content_type not in {"text/html", "application/xhtml+xml"}:
        raise FetchError("The source did not return an HTML article page.")
    return content, content_type, final_url


def extract_article(url: str) -> dict[str, object]:
    content, _, final_url = fetch_bytes(url)
    soup = BeautifulSoup(content, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        if node.name == "script" and node.get("type") == "application/ld+json":
            continue
        node.decompose()

    def meta(*names: str) -> str | None:
        for name in names:
            node = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if node and node.get("content"):
                value = " ".join(unescape(str(node["content"])).split())
                if value:
                    return value[:1000]
        return None

    headline = meta("og:title", "twitter:title")
    if not headline and soup.title:
        headline = " ".join(soup.title.get_text(" ", strip=True).split())[:1000]
    canonical_node = soup.find("link", rel=lambda rel: rel and "canonical" in rel)
    canonical = urljoin(final_url, str(canonical_node.get("href"))) if canonical_node and canonical_node.get("href") else final_url
    page_title = headline
    body_node = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [" ".join(p.get_text(" ", strip=True).split()) for p in body_node.find_all(["p", "h2", "h3"])]
    body = "\n\n".join(part for part in paragraphs if len(part) >= 30)[:100_000]
    warnings = []
    live_blog = "/liveblog/" in urlsplit(final_url).path.lower() or "live blog" in (page_title or "").lower()
    selected_segment = _live_segment(body_node) if live_blog else None
    if live_blog:
        warnings.append("This is a live or continuously updated page; content may change after extraction.")
        if selected_segment:
            headline = str(selected_segment["headline"])
            body = str(selected_segment["body"])
        else:
            warnings.append("No coherent live-update text/image segment was found; automatic multimodal pairing is disabled.")
    publisher = meta("og:site_name", "publisher")
    published = meta("article:published_time", "datePublished", "pubdate")
    if not headline:
        warnings.append("No headline metadata was found.")
    if not publisher:
        warnings.append("No publisher metadata was found.")
    if not published:
        warnings.append("No publication date metadata was found.")
    if not body:
        body = " ".join(body_node.get_text(" ", strip=True).split())[:100_000]
        if body:
            warnings.append("Article body structure was unclear; extracted visible page text.")
    if not body:
        warnings.append("No article body text could be extracted.")
    images = []
    candidates: list[dict[str, object]] = []

    def add_candidate(
        source_url: object,
        source_kind: str,
        *,
        alt: str | None = None,
        caption: str | None = None,
        width: object = None,
        height: object = None,
        body_position: int | None = None,
    ) -> None:
        if not isinstance(source_url, str) or not source_url.strip():
            return
        candidates.append(
            {
                "source_url": source_url,
                "source_kind": source_kind,
                "alt": alt,
                "caption": caption,
                "width": width,
                "height": height,
                "body_position": body_position,
            }
        )

    for key in ("og:image", "twitter:image"):
        found = meta(key)
        if found:
            prefix = key.rsplit(":", 1)[0]
            add_candidate(
                found,
                key,
                width=meta(f"{prefix}:image:width"),
                height=meta(f"{prefix}:image:height"),
            )
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entries.extend(
                child for child in entry.get("@graph", []) if isinstance(child, dict)
            )
            article_type = entry.get("@type", [])
            if isinstance(article_type, str):
                article_type = [article_type]
            if not isinstance(article_type, list) or not any(
                str(value).rsplit("/", 1)[-1] in {"Article", "NewsArticle"}
                for value in article_type
            ):
                continue
            image_value = entry.get("image")
            if isinstance(image_value, list):
                image_value = image_value[0] if image_value else None
            width = height = None
            if isinstance(image_value, dict):
                width = image_value.get("width")
                height = image_value.get("height")
                image_value = image_value.get("url") or image_value.get("contentUrl")
            add_candidate(image_value, "structured_data", width=width, height=height)
    for position, node in enumerate(body_node.find_all("img")):
        src = node.get("src") or node.get("data-src") or node.get("data-lazy-src")
        alt = " ".join(str(node.get("alt", "")).split())[:500] or None
        if src:
            caption_node = node.find_parent("figure")
            caption = None
            if caption_node and caption_node.find("figcaption"):
                caption = caption_node.find("figcaption").get_text(" ", strip=True)[:500]
            add_candidate(
                str(src),
                "article_body",
                alt=alt,
                caption=caption,
                width=node.get("width"),
                height=node.get("height"),
                body_position=position,
            )
    seen = set()
    segment_image = urljoin(final_url, str(selected_segment["image_url"])) if selected_segment else None
    for candidate in candidates:
        src = str(candidate["source_url"])
        alt = candidate["alt"]
        caption = candidate["caption"]
        source_kind = str(candidate["source_kind"])
        absolute = urljoin(final_url, src)
        try:
            public_url(absolute)
        except FetchError:
            continue
        parsed_path = urlsplit(absolute).path.lower()
        if any(term in parsed_path for term in ("logo", "icon", "avatar", "sprite", "tracking", "pixel")):
            continue
        if live_blog and selected_segment and absolute != segment_image:
            continue
        if absolute not in seen:
            seen.add(absolute)
            score = {"structured_data": 100, "og:image": 90, "twitter:image": 80, "article_body": 45}[source_kind]
            if live_blog and selected_segment:
                score += 20
                reasons = ["same live-update segment"]
            else:
                reasons = [source_kind.replace("_", " ")]
            try:
                width, height = int(candidate["width"] or 0), int(candidate["height"] or 0)
            except (TypeError, ValueError):
                width = height = 0
            if width >= 300 and height >= 200:
                score += 8
                reasons.append("declared editorial dimensions")
            if caption:
                score += 10
                reasons.append("caption")
            if alt and len(alt) > 12:
                score += 5
                reasons.append("descriptive alt text")
            if source_kind == "article_body" and candidate["body_position"] is not None:
                score += max(0, 15 - int(candidate["body_position"]) * 3)
                reasons.append("article proximity")
            images.append(
                {
                    "source_url": absolute,
                    "alt_text": alt,
                    "caption": caption,
                    "source_kind": source_kind,
                    "ranking_score": score,
                    "ranking_reason": ", ".join(reasons),
                }
            )
    images.sort(key=lambda item: int(item["ranking_score"]), reverse=True)
    images = images[:8]
    if not images:
        warnings.append("No suitable article images were found; add an image manually to analyze a pair.")
    primary_index = 0 if images and int(images[0]["ranking_score"]) >= 70 and (not live_blog or selected_segment) else None
    if images and primary_index is None:
        warnings.append("Image candidates were found, but metadata confidence is low; select one manually.")
    return {
        "submitted_url": url,
        "canonical_url": canonical,
        "page_title": page_title,
        "is_live_blog": live_blog,
        "selected_segment": selected_segment,
        "headline": headline,
        "body": body,
        "publisher": publisher,
        "published_at": published,
        "warnings": warnings,
        "images": images,
        "primary_image_index": primary_index,
        "primary_image_reason": images[0]["ranking_reason"] if primary_index is not None else None,
        "content_hash": hashlib.sha256(((headline or "") + "\n" + body).encode("utf-8")).hexdigest(),
    }
