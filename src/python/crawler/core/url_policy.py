"""Generic URL canonicalization and crawl-trap policy."""
from __future__ import annotations

import posixpath
import re
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit


TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "mc_cid",
        "mc_eid",
        "ref_src",
        "igshid",
    }
)
ACTION_PARAMETERS = frozenset(
    {
        "action",
        "veaction",
        "printable",
        "download",
        "logout",
        "login",
        "returnto",
    }
)
ACTION_VALUES = frozenset(
    {"edit", "history", "delete", "login", "logout", "submit", "createaccount"}
)
ACTION_PATH = re.compile(
    r"/(?:login|logout|signin|signup|register|cart|checkout|admin|special:"
    r"(?:userlogin|createaccount|editpage|downloadaspdf))(?=/|$)",
    re.IGNORECASE,
)
CALENDAR_PATH = re.compile(r"/(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?:/|$)")


def canonicalize_url(url: str, base_url: str = "") -> str:
    if not url:
        return ""
    try:
        absolute = urljoin(base_url, str(url).strip())
        parsed = urlsplit(absolute)
    except ValueError:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return ""
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        return ""
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    raw_path = parsed.path or "/"
    # Preserve existing escapes while encoding Unicode/control/space characters.
    encoded_path = quote(raw_path, safe="/%:@!$&'()*+,;=-._~")
    normalized_path = posixpath.normpath(encoded_path)
    if encoded_path.endswith("/") and not normalized_path.endswith("/"):
        normalized_path += "/"
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    try:
        parameters = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=50)
    except ValueError:
        return ""
    filtered = []
    for key, value in parameters:
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_PARAMETERS:
            continue
        filtered.append((key, value))
    filtered.sort(key=lambda pair: (pair[0], pair[1]))
    return urlunsplit((scheme, netloc, normalized_path, urlencode(filtered, doseq=True), ""))


def rejection_reason(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        parameters = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=50)
    except ValueError:
        return "invalid_url"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "unsupported_url"
    if len(parameters) > 10:
        return "query_trap"
    for key, value in parameters:
        lowered_key = key.lower()
        lowered_value = value.lower()
        if lowered_key in ACTION_PARAMETERS and (
            lowered_key != "action" or lowered_value in ACTION_VALUES
        ):
            return "action_url"
        if lowered_value in ACTION_VALUES:
            return "action_url"
    if ACTION_PATH.search(parsed.path):
        return "action_url"
    if CALENDAR_PATH.search(parsed.path):
        return "calendar_trap"
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) > 20 or any(
        segments[index : index + 3] == [segments[index]] * 3
        for index in range(max(0, len(segments) - 2))
    ):
        return "path_trap"
    return None


def is_crawlable(url: str) -> bool:
    return rejection_reason(url) is None
