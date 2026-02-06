import json
import os
import random
import re
import time
from html.parser import HTMLParser
from urllib.parse import urlparse
from pathlib import Path

import requests

# Online product URLs to scrape
DEFAULT_URLS = [
    "https://www.ebay.com/itm/276265496033?_skw=gaming+monitor&hash=item4052b4ede1:g:51cAAeSwDXRpOygb",
    "https://www.ebay.com/itm/226964309950?_skw=gaming+monitor&hash=item34d82087be:g:9~QAAeSw1e5ovvdX",
    "https://www.ebay.com/itm/256853107342?_skw=gaming+monitor&hash=item3bcda3628e:g:vNoAAeSwzoRpVB7~",
    "https://www.ebay.com/itm/256547414823?_skw=gaming+monitor&hash=item3bbb6ae327:g:OvQAAeSw4UVpG3u5",
    # add more URLs here
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}

EBAY_JSON_DIR = os.path.join(
    "notebook",
    "data",
    "ebay",
    os.getenv("EBAY_QUERY", "gaming_monitor").replace(" ", "_").lower(),
)
USE_DEFAULT_URLS = os.getenv("USE_DEFAULT_URLS", "0") == "1"
SAFE_MODE = os.getenv("SAFE_MODE", "1") == "1"
STOP_ON_BLOCK = os.getenv("STOP_ON_BLOCK", "1") == "1"
USE_CACHE = os.getenv("USE_CACHE", "1") == "1"
CACHE_DIR = os.getenv("CACHE_DIR", "notebook/data/html_cache")
MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "0"))  # 0 = no limit
LONG_BREAK_EVERY = int(os.getenv("LONG_BREAK_EVERY", "10"))
LONG_BREAK_RANGE = (300, 900)  # 5-15 minutes
ONE_PER_MINUTE = os.getenv("ONE_PER_MINUTE", "0") == "1"
SAVE_BLOCKED_HTML = os.getenv("SAVE_BLOCKED_HTML", "1") == "1"
SHOW_BLOCKED_SNIPPET = os.getenv("SHOW_BLOCKED_SNIPPET", "0") == "1"

MAX_RETRIES = 2
BASE_DELAY_SECONDS = 20.0
JITTER_RANGE = (0.8, 1.8)
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT = 25
BLOCK_COOLDOWN_SECONDS = (120, 300)
SAFE_DELAY_SECONDS = (30, 60)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.1:8b-instruct")


class VisibleTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return "\n".join(self._parts)


def _cache_path_for_url(url):
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")
    if len(safe_name) > 180:
        safe_name = safe_name[:180]
    return Path(CACHE_DIR) / f"{safe_name}.html"


def fetch_html(session, url):
    if USE_CACHE:
        cache_path = _cache_path_for_url(url)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if response.status_code in {429, 403, 500, 502, 503, 504}:
                raise requests.RequestException(
                    f"Status {response.status_code} for {url}"
                )
            response.raise_for_status()
            html = response.text
            if _looks_blocked(html):
                if SAVE_BLOCKED_HTML:
                    blocked_dir = Path(CACHE_DIR) / "blocked"
                    blocked_dir.mkdir(parents=True, exist_ok=True)
                    blocked_path = blocked_dir / f"{int(time.time())}.html"
                    blocked_path.write_text(html, encoding="utf-8")
                    print(f"Blocked page saved to {blocked_path}")
                if SHOW_BLOCKED_SNIPPET:
                    snippet = re.sub(r"\\s+", " ", html[:1000]).strip()
                    print(f"Blocked page snippet: {snippet}")
                cooldown = random.uniform(*BLOCK_COOLDOWN_SECONDS)
                print(f"Blocked page detected. Cooling down for {cooldown:.1f}s...")
                time.sleep(cooldown)
                raise requests.RequestException("Blocked page detected")
            if USE_CACHE:
                Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
                _cache_path_for_url(url).write_text(html, encoding="utf-8")
            return html
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise Exception(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {exc}")
            sleep_for = BASE_DELAY_SECONDS * (BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(
                *JITTER_RANGE
            )
            print(f"Attempt {attempt} failed for {url}: {exc}")
            print(f"Sleeping {sleep_for:.1f}s before retry...")
            time.sleep(sleep_for)


def _looks_blocked(html):
    if not html:
        return True
    lowered = html.lower()
    return (
        "pardon our interruption" in lowered
        or "checking your browser before you access ebay" in lowered
        or "your browser will redirect to your requested content shortly" in lowered
        or "reference id:" in lowered
        or "unusual traffic" in lowered
        or "access denied" in lowered
        or "captcha" in lowered
    )


def extract_title(html):
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or None


def infer_brand_model_from_title(title):
    if not title:
        return None, None
    cleaned = re.sub(r"\s+", " ", title).strip()
    tokens = cleaned.split()
    if not tokens:
        return None, None
    brand = tokens[0]
    model = None
    for token in tokens[1:6]:
        if re.search(r"\d", token):
            model = token
            break
    return brand, model


def extract_json_ld(html):
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    data = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                data.extend(parsed)
            else:
                data.append(parsed)
        except json.JSONDecodeError:
            continue
    return data


def extract_visible_text(html):
    parser = VisibleTextExtractor()
    parser.feed(html)
    text = parser.get_text()
    return re.sub(r"[ \t]+", " ", text).strip()

def load_urls_from_ebay_json(dir_path):
    urls = []
    path = Path(dir_path)
    if not path.exists():
        return urls
    for json_file in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data.get("itemSummaries", [])
        for item in items:
            url = item.get("itemWebUrl")
            if url and url.startswith("http"):
                urls.append(url)
    # de-duplicate preserving order
    seen = set()
    deduped = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def extract_kv_pairs_from_html(html):
    pairs = []
    # Table rows: <tr><th>Key</th><td>Value</td>
    for match in re.findall(
        r"<tr[^>]*>\s*<t[hd][^>]*>(.*?)</t[hd]>\s*<t[hd][^>]*>(.*?)</t[hd]>\s*</tr>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        key = re.sub(r"<[^>]+>", " ", match[0])
        val = re.sub(r"<[^>]+>", " ", match[1])
        key = re.sub(r"\s+", " ", key).strip()
        val = re.sub(r"\s+", " ", val).strip()
        if key and val:
            pairs.append((key, val))

    # Definition lists: <dt>Key</dt><dd>Value</dd>
    for match in re.findall(
        r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        key = re.sub(r"<[^>]+>", " ", match[0])
        val = re.sub(r"<[^>]+>", " ", match[1])
        key = re.sub(r"\s+", " ", key).strip()
        val = re.sub(r"\s+", " ", val).strip()
        if key and val:
            pairs.append((key, val))

    # List items with colon: <li>Key: Value</li>
    for match in re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", match)
        text = re.sub(r"\s+", " ", text).strip()
        if ":" in text:
            key, val = text.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                pairs.append((key, val))

    return pairs

def extract_relevant_text(text, max_chars=6000):
    keywords = re.compile(
        r"(inch|\"|hz|ips|va|tn|oled|resolution|hdmi|displayport|usb|nits|brightness|"
        r"contrast|response|ms|vesa|hdr|sync|g-sync|freesync|adaptive|refresh|panel|"
        r"sRGB|dci-p3|color|speaker|audio|curved|curvature|aspect|ratio)",
        re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [line.strip() for line in re.split(r"[.!?]\s+", text) if line.strip()]
    scored = [line for line in lines if keywords.search(line)]
    fallback = lines[:200]
    combined = []
    seen = set()
    for line in scored + fallback:
        if line in seen:
            continue
        seen.add(line)
        combined.append(line)
        if sum(len(item) + 1 for item in combined) >= max_chars:
            break
    return "\n".join(combined)[:max_chars]


def build_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_url",
            "source_domain",
            "title",
            "brand",
            "model",
            "color",
            "screen_size_in",
            "resolution",
            "refresh_rate_hz",
            "panel_type",
            "aspect_ratio",
            "response_time_ms",
            "brightness_nits",
            "hdr",
            "has_adaptive_sync",
            "ports",
            "dimensions_in",
        ],
        "properties": {
            "source_url": {"type": "string"},
            "source_domain": {"type": ["string", "null"]},
            "title": {"type": ["string", "null"]},
            "brand": {"type": ["string", "null"]},
            "model": {"type": ["string", "null"]},
            "color": {"type": ["string", "null"]},
            "screen_size_in": {"type": ["number", "null"]},
            "resolution": {
                "type": "object",
                "additionalProperties": False,
                "required": ["width", "height", "label"],
                "properties": {
                    "width": {"type": ["integer", "null"]},
                    "height": {"type": ["integer", "null"]},
                    "label": {"type": ["string", "null"]},
                },
            },
            "refresh_rate_hz": {"type": ["number", "null"]},
            "panel_type": {"type": ["string", "null"]},
            "aspect_ratio": {"type": ["string", "null"]},
            "response_time_ms": {"type": ["number", "null"]},
            "brightness_nits": {"type": ["number", "null"]},
            "hdr": {"type": ["string", "null"]},
            "has_adaptive_sync": {"type": ["boolean", "null"]},
            "ports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "count", "version"],
                    "properties": {
                        "type": {"type": ["string", "null"]},
                        "count": {"type": ["integer", "null"]},
                        "version": {"type": ["string", "null"]},
                    },
                },
            },
            "dimensions_in": {
                "type": "object",
                "additionalProperties": False,
                "required": ["width", "height", "depth"],
                "properties": {
                    "width": {"type": ["number", "null"]},
                    "height": {"type": ["number", "null"]},
                    "depth": {"type": ["number", "null"]},
                },
            },
        },
    }


def empty_result(url, title):
    source_domain = urlparse(url).netloc if url else None
    return {
        "source_url": url or "",
        "source_domain": source_domain,
        "title": title,
        "brand": None,
        "model": None,
        "color": None,
        "screen_size_in": None,
        "resolution": {"width": None, "height": None, "label": None},
        "refresh_rate_hz": None,
        "panel_type": None,
        "aspect_ratio": None,
        "response_time_ms": None,
        "brightness_nits": None,
        "hdr": None,
        "has_adaptive_sync": None,
        "ports": [],
        "dimensions_in": {"width": None, "height": None, "depth": None},
    }


def normalize_number(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_resolution(text):
    match = re.search(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})", text)
    if not match:
        alias = text.strip().lower()
        if "full hd" in alias or "fhd" in alias:
            return {"width": 1920, "height": 1080, "label": "1920x1080"}
        if "qhd" in alias or "1440p" in alias:
            return {"width": 2560, "height": 1440, "label": "2560x1440"}
        if "uhd" in alias or "4k" in alias:
            return {"width": 3840, "height": 2160, "label": "3840x2160"}
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    label = f"{width}x{height}"
    return {"width": width, "height": height, "label": label}


def parse_screen_size_in(text):
    match = re.search(r"(\d{2}(\.\d+)?)\s*(\"|inch|inches|in)\b", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def parse_refresh_rate(text):
    match = re.search(r"(\d{2,3}(\.\d+)?)\s*hz", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def parse_response_time(text):
    match = re.search(r"(\d+(\.\d+)?)\s*ms", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def parse_brightness(text):
    match = re.search(r"(\d{2,4}(\.\d+)?)\s*nits", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def parse_panel_type(text):
    match = re.search(r"\b(IPS|VA|TN|OLED|Mini-LED|QLED)\b", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def parse_hdr(text):
    match = re.search(r"\b(HDR\d{2,3}|HDR|DisplayHDR\s*\d{2,4})\b", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(0).strip()


def parse_adaptive_sync(text):
    match = re.search(r"\b(FreeSync|G-Sync|Adaptive[- ]Sync)\b", text, re.IGNORECASE)
    if not match:
        return None
    return True


def parse_color(text):
    match = re.search(r"\b(black|white|silver|gray|grey|red|blue|green|gold)\b", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).capitalize()


def parse_dimension_in(text):
    match = re.search(r"(\d{1,3}(\.\d+)?)\s*(in|inch|inches|\"|''|”)?\b", text, re.IGNORECASE)
    if not match:
        numeric = re.search(r"^\s*(\d{1,3}(\.\d+)?)\s*$", text)
        if numeric:
            return float(numeric.group(1))
        return None
    return float(match.group(1))


def normalize_key(key):
    key = key.strip().lower()
    key = re.sub(r"[\s_]+", " ", key)
    return key


def apply_kv_pairs(base, pairs):
    for raw_key, raw_val in pairs:
        key = normalize_key(raw_key)
        val = raw_val.strip()

        if key in {"brand", "manufacturer", "make"} and base["brand"] is None:
            base["brand"] = val
        elif key in {"model", "model number", "model name", "mpn"} and base["model"] is None:
            base["model"] = val
        elif key in {"color", "colour"} and base["color"] is None:
            base["color"] = val
        elif key in {"screen size", "display size", "size"} and base["screen_size_in"] is None:
            size = parse_screen_size_in(val)
            if size is not None:
                base["screen_size_in"] = size
        elif key in {"resolution", "display resolution", "screen resolution", "maximum resolution", "native resolution"} and base["resolution"]["label"] is None:
            res = parse_resolution(val)
            if res:
                base["resolution"] = res
        elif key in {"refresh rate", "refresh rate (max)"} and base["refresh_rate_hz"] is None:
            rate = parse_refresh_rate(val)
            if rate is not None:
                base["refresh_rate_hz"] = rate
        elif key in {"panel type", "panel", "display screen technology", "display type"} and base["panel_type"] is None:
            panel = parse_panel_type(val) or val
            base["panel_type"] = panel
        elif key in {"response time"} and base["response_time_ms"] is None:
            resp = parse_response_time(val)
            if resp is not None:
                base["response_time_ms"] = resp
        elif key in {"brightness", "brightness (typical)"} and base["brightness_nits"] is None:
            bright = parse_brightness(val)
            if bright is not None:
                base["brightness_nits"] = bright
        elif key in {"aspect ratio"} and base["aspect_ratio"] is None:
            base["aspect_ratio"] = val
        elif key in {"hdr"} and base["hdr"] is None:
            base["hdr"] = val
        elif key in {"adaptive sync", "adaptive-sync", "freesync", "g-sync"} and base["has_adaptive_sync"] is None:
            base["has_adaptive_sync"] = True
        elif key in {"item width", "width"} and base["dimensions_in"]["width"] is None:
            dim = parse_dimension_in(val)
            if dim is not None:
                base["dimensions_in"]["width"] = dim
        elif key in {"item height", "height"} and base["dimensions_in"]["height"] is None:
            dim = parse_dimension_in(val)
            if dim is not None:
                base["dimensions_in"]["height"] = dim
        elif key in {"item depth", "depth"} and base["dimensions_in"]["depth"] is None:
            dim = parse_dimension_in(val)
            if dim is not None:
                base["dimensions_in"]["depth"] = dim
        elif key in {"ports", "inputs"} and not base["ports"]:
            base["ports"] = parse_ports(val)
    return base


def parse_ports(text):
    ports = []
    for port_type in ["HDMI", "DisplayPort", "USB-C", "USB", "Thunderbolt"]:
        pattern = rf"(\d+)\s*x?\s*{re.escape(port_type)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ports.append({"type": port_type, "count": int(match.group(1)), "version": None})
    return ports


def select_primary_product(json_ld_items):
    products = []
    for item in json_ld_items:
        if isinstance(item, dict) and "@graph" in item:
            graph = item.get("@graph", [])
            if isinstance(graph, list):
                products.extend(graph)
        else:
            products.append(item)

    candidates = []
    for item in products:
        if not isinstance(item, dict):
            continue
        item_type = item.get("@type")
        if isinstance(item_type, list):
            if "Product" not in item_type:
                continue
        elif item_type != "Product":
            continue
        candidates.append(item)

    if not candidates:
        return None

    def score(product):
        score_value = 0
        if product.get("name"):
            score_value += 2
        if product.get("offers"):
            score_value += 2
        if product.get("brand"):
            score_value += 1
        return score_value

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def extract_from_json_ld(product):
    data = empty_result(None, None)
    if not product:
        return data

    data["title"] = product.get("name") or product.get("description")

    brand = product.get("brand")
    if isinstance(brand, dict):
        data["brand"] = brand.get("name")
    elif isinstance(brand, str):
        data["brand"] = brand

    data["model"] = product.get("model") or product.get("mpn")
    if data["brand"] is None or data["model"] is None:
        inferred_brand, inferred_model = infer_brand_model_from_title(data["title"])
        if data["brand"] is None:
            data["brand"] = inferred_brand
        if data["model"] is None:
            data["model"] = inferred_model

    additional = product.get("additionalProperty")
    if isinstance(additional, list):
        pairs = []
        for item in additional:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            value = item.get("value")
            if name and value:
                pairs.append((str(name), str(value)))
        data = apply_kv_pairs(data, pairs)

    return data


def merge_field(base, key, value):
    if base.get(key) is None and value is not None:
        base[key] = value


def fill_from_text(base, text):
    pairs = []
    # Line-based key: value patterns
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                pairs.append((key, val))
    if pairs:
        base = apply_kv_pairs(base, pairs)

    merge_field(base, "screen_size_in", parse_screen_size_in(text))
    if base["resolution"]["label"] is None:
        res = parse_resolution(text)
        if res:
            base["resolution"] = res
    merge_field(base, "refresh_rate_hz", parse_refresh_rate(text))
    merge_field(base, "panel_type", parse_panel_type(text))
    merge_field(base, "response_time_ms", parse_response_time(text))
    merge_field(base, "brightness_nits", parse_brightness(text))
    merge_field(base, "hdr", parse_hdr(text))
    merge_field(base, "has_adaptive_sync", parse_adaptive_sync(text))
    merge_field(base, "color", parse_color(text))
    if base["resolution"]["label"] is None:
        res = parse_resolution(text)
        if res:
            base["resolution"] = res
    if not base["ports"]:
        base["ports"] = parse_ports(text)
    return base


def pick_ollama_model():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        models = [item.get("name") for item in response.json().get("models", [])]
        if OLLAMA_MODEL in models:
            return OLLAMA_MODEL
        if OLLAMA_FALLBACK_MODEL in models:
            return OLLAMA_FALLBACK_MODEL
    except Exception:
        return None
    return None


def llm_cleanup(stage1_data, relevant_text, title, url):
    model_name = pick_ollama_model()
    if not model_name:
        return stage1_data

    prompt = (
        "You are a normalization assistant. Only fill null fields using explicit statements "
        "from the provided text. Do not guess or estimate. Do not change non-null fields. "
        "Normalize units (inches, Hz, ms, nits, W, mm, kg). Return JSON only. "
        "Only include these fields: brand, model, screen_size_in, resolution, refresh_rate_hz, "
        "panel_type, response_time_ms, brightness_nits, hdr, has_adaptive_sync, aspect_ratio, ports, "
        "color, dimensions_in.width, dimensions_in.height, dimensions_in.depth."
    )
    payload = {
        "model": model_name,
        "prompt": json.dumps(
            {
                "instructions": prompt,
                "schema": build_schema(),
                "current_data": stage1_data,
                "visible_text": relevant_text,
                "title": title,
                "url": url,
            },
            ensure_ascii=True,
        ),
        "temperature": 0.1,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        output = response.json().get("response", "").strip()
        return json.loads(output or "{}")
    except Exception:
        return stage1_data


def parse_html_with_agent(url, html):
    title = extract_title(html)
    json_ld = extract_json_ld(html)
    visible_text = extract_visible_text(html)
    relevant_text = extract_relevant_text(visible_text)
    source_domain = urlparse(url).netloc if url else None
    kv_pairs = extract_kv_pairs_from_html(html)

    product = select_primary_product(json_ld)
    data = extract_from_json_ld(product)
    if data["title"] is None:
        data["title"] = title

    if product is None:
        data = empty_result(url, title)

    if kv_pairs:
        data = apply_kv_pairs(data, kv_pairs)
    data = fill_from_text(data, relevant_text)
    data = llm_cleanup(data, relevant_text, title, url)

    data["source_url"] = url or ""
    data["source_domain"] = source_domain
    data["title"] = data.get("title") or title
    return data


def write_results(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_existing_urls(file_path):
    if not os.path.exists(file_path):
        return set()
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except Exception:
        return set()
    urls = set()
    if isinstance(data, list):
        for item in data:
            url = item.get("source_url")
            if url:
                urls.add(url)
    return urls


def main():
    raise RuntimeError(
        "Online scraping has been disabled. Use src/components/monitor_agent_offline.py for offline parsing."
    )


if __name__ == "__main__":
    main()
