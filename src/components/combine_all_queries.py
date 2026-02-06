import os
from pathlib import Path

import pandas as pd
from urllib.parse import urlparse


BASE_DIR = Path("notebook/data/ebay")
OUTPUT_CSV = BASE_DIR / "all_queries.csv"
INCLUDE_QUERY_COL = True


def _find_combined_files(base_dir: Path):
    if not base_dir.exists():
        return []
    files = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        combined_path = child / "combined_training.csv"
        if combined_path.exists():
            files.append((child.name, combined_path))
    return files


def _normalize_url(url: str | None):
    if not url or not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _dedupe(df: pd.DataFrame):
    if "item_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["item_id"], keep="first")
        after = len(df)
        if before != after:
            print(f"Deduped by item_id: {before} -> {after}")
        return df

    if "item_url" in df.columns:
        df = df.copy()
        df["_normalized_url"] = df["item_url"].apply(_normalize_url)
        before = len(df)
        df = df.drop_duplicates(subset=["_normalized_url"], keep="first")
        after = len(df)
        if before != after:
            print(f"Deduped by item_url: {before} -> {after}")
        df = df.drop(columns=["_normalized_url"])
        return df

    return df


def main():
    files = _find_combined_files(BASE_DIR)
    if not files:
        raise FileNotFoundError(f"No combined_training.csv files found under {BASE_DIR}")

    frames = []
    for query_slug, path in files:
        df = pd.read_csv(path)
        if INCLUDE_QUERY_COL and "query_slug" not in df.columns:
            df.insert(0, "query_slug", query_slug)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = _dedupe(combined)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"Found {len(files)} query folders.")
    print(f"Saved combined dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
