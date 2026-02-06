import json
import os
import re
from pathlib import Path

from src.components.monitor_agent import parse_html_with_agent, write_results

QUERY_SLUG = "budget_monitor" # check product_ebay.py
HTML_DIR = f"notebook/data/ebay/{QUERY_SLUG}/html_cache"
OUTPUT_JSON = f"notebook/data/ebay/{QUERY_SLUG}/monitors.json"
MAX_PER_RUN = 0  # 0 = no limit


def _find_canonical_url(html):
    match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def _load_existing_urls(file_path):
    if not os.path.exists(file_path):
        return set(), []
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except Exception:
        return set(), []
    if not isinstance(data, list):
        return set(), []
    return {item.get("source_url") for item in data if item.get("source_url")}, data


def main():
    html_dir = Path(HTML_DIR)
    if not html_dir.exists():
        raise FileNotFoundError(f"HTML_DIR not found: {HTML_DIR}")

    existing_urls, existing_data = _load_existing_urls(OUTPUT_JSON)
    html_files = sorted(html_dir.glob("*.html"))
    results = []

    for idx, path in enumerate(html_files):
        if MAX_PER_RUN and idx >= MAX_PER_RUN:
            print(f"Reached MAX_PER_RUN={MAX_PER_RUN}. Stopping early.")
            break

        html = path.read_text(encoding="utf-8")
        canonical = _find_canonical_url(html)
        source_url = canonical or path.as_posix()

        if source_url in existing_urls:
            continue

        print(f"Parsing ({idx + 1}/{len(html_files)}): {path}")
        json_data = parse_html_with_agent(source_url, html)
        results.append(json_data)

    if results:
        merged = existing_data + results
        write_results(merged, OUTPUT_JSON)
        print(f"Saved parsed monitor data to {OUTPUT_JSON}")
    else:
        print("No new HTML files to parse.")


if __name__ == "__main__":
    main()
