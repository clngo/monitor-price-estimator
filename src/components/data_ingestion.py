import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import get_latest_json, save_object


@dataclass
class DataIngestionConfig:
    all_queries_csv: str = os.path.join(
        "notebook",
        "data",
        "ebay",
        "all_queries.csv",
    )
    monitors_json: str = os.path.join(
        "notebook",
        "data",
        "ebay",
        os.getenv("EBAY_QUERY", "gaming_monitor").replace(" ", "_").lower(),
        "monitors.json",
    )
    listings_dir: str = os.path.join(
        "notebook",
        "data",
        "ebay",
        os.getenv("EBAY_QUERY", "gaming_monitor").replace(" ", "_").lower(),
    )
    listings_json: str | None = None
    output_csv: str = os.path.join(
        "notebook",
        "data",
        "ebay",
        os.getenv("EBAY_QUERY", "gaming_monitor").replace(" ", "_").lower(),
        "combined_training.csv",
    )
    output_pkl: str = os.path.join(
        "notebook",
        "data",
        "ebay",
        os.getenv("EBAY_QUERY", "gaming_monitor").replace(" ", "_").lower(),
        "combined_training.pkl",
    )
    train_data_path: str = os.path.join("data_private", "train.csv")
    test_data_path: str = os.path.join("data_private", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(
        self,
        monitors_json: str | None = None,
        listings_json: str | None = None,
    ):
        logging.info("Entered the data ingestion pipeline.")

        try:
            all_queries_path = self.ingestion_config.all_queries_csv
            if os.path.exists(all_queries_path):
                logging.info("Using aggregated dataset: %s", all_queries_path)
                df = pd.read_csv(all_queries_path)
            else:
                logging.info("Building dataset from monitor + listings merge.")
                monitors_path = monitors_json or self.ingestion_config.monitors_json
                if listings_json:
                    listings_path = listings_json
                else:
                    listings_path = get_latest_json(self.ingestion_config.listings_dir, pattern="listings_*.json")

                monitors = self._load_monitors(monitors_path)
                listings = self._load_listings(listings_path)
                combined = self._join_records(monitors, listings)
                if not combined:
                    raise ValueError("No matching records found between monitor specs and listings.")

                df = pd.DataFrame(combined)
                output_csv = self.ingestion_config.output_csv
                output_pkl = self.ingestion_config.output_pkl

                Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_csv, index=False)
                save_object(output_pkl, df)

                logging.info("Saved combined training data to %s", output_csv)
                logging.info("Saved combined training dataframe to %s", output_pkl)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_path = self.ingestion_config.train_data_path
            test_path = self.ingestion_config.test_data_path
            Path(train_path).parent.mkdir(parents=True, exist_ok=True)
            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)
            logging.info("Train/test split completed: %s | %s", train_path, test_path)

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)

    def _normalize_url(self, url):
        if not url:
            return None
        parsed = urlparse(url)
        if not parsed.scheme:
            return url
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _extract_legacy_item_id(self, url):
        if not url:
            return None
        match = re.search(r"/itm/(\\d+)", url)
        if match:
            return match.group(1)
        return None

    def _parse_item_id(self, item_id):
        if not item_id:
            return None
        match = re.search(r"\\|(\\d+)\\|", item_id)
        if match:
            return match.group(1)
        return None

    def _safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return None

    def _parse_date(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            return None

    def _extract_shipping(self, shipping_options):
        if not shipping_options:
            return {
                "shipping_cost": None,
                "shipping_currency": None,
                "shipping_cost_type": None,
                "min_delivery_date": None,
                "max_delivery_date": None,
            }
        first = shipping_options[0]
        cost = first.get("shippingCost", {})
        return {
            "shipping_cost": self._safe_float(cost.get("value")),
            "shipping_currency": cost.get("currency"),
            "shipping_cost_type": first.get("shippingCostType"),
            "min_delivery_date": self._parse_date(first.get("minEstimatedDeliveryDate")),
            "max_delivery_date": self._parse_date(first.get("maxEstimatedDeliveryDate")),
        }

    def _load_monitors(self, path):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Monitors JSON not found: {path}")
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("items", [])
        monitors = []
        for entry in data:
            url = entry.get("source_url")
            monitors.append(
                {
                    **entry,
                    "_normalized_url": self._normalize_url(url),
                    "_legacy_item_id": self._extract_legacy_item_id(url),
                }
            )
        return monitors

    def _load_listings(self, path):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Listings JSON not found: {path}")
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        items = data.get("itemSummaries", [])
        listings = []
        for item in items:
            item_url = item.get("itemWebUrl")
            listings.append(
                {
                    "item_id": item.get("itemId"),
                    "legacy_item_id": item.get("legacyItemId") or self._parse_item_id(item.get("itemId")),
                    "item_url": item_url,
                    "_normalized_url": self._normalize_url(item_url),
                    "title": item.get("title"),
                    "price_value": self._safe_float(item.get("price", {}).get("value")),
                    "price_currency": item.get("price", {}).get("currency"),
                    "condition": item.get("condition"),
                    "condition_id": item.get("conditionId"),
                    "seller_username": (item.get("seller") or {}).get("username"),
                    "seller_feedback_pct": self._safe_float((item.get("seller") or {}).get("feedbackPercentage")),
                    "seller_feedback_score": self._safe_float((item.get("seller") or {}).get("feedbackScore")),
                    "buying_options": ",".join(item.get("buyingOptions", []) or []),
                    **self._extract_shipping(item.get("shippingOptions", [])),
                }
            )
        return listings

    def _join_records(self, monitors, listings):
        by_url = self._build_best_index(monitors, "_normalized_url")
        by_legacy = self._build_best_index(monitors, "_legacy_item_id")
        by_title = self._build_best_index(monitors, "_title_key")

        combined = []
        for listing in listings:
            monitor = None
            if listing.get("_normalized_url") and listing["_normalized_url"] in by_url:
                monitor = by_url[listing["_normalized_url"]]
            elif listing.get("legacy_item_id") and listing["legacy_item_id"] in by_legacy:
                monitor = by_legacy[listing["legacy_item_id"]]
            else:
                title_key = self._normalize_title(listing.get("title"))
                if title_key and title_key in by_title:
                    monitor = by_title[title_key]

            if not monitor:
                continue

            combined.append(
                {
                    "item_id": listing.get("item_id"),
                    "legacy_item_id": listing.get("legacy_item_id"),
                    "item_url": listing.get("item_url"),
                    "title": listing.get("title"),
                    "price_value": listing.get("price_value"),
                    "price_currency": listing.get("price_currency"),
                    "condition": listing.get("condition"),
                    "condition_id": listing.get("condition_id"),
                    "seller_username": listing.get("seller_username"),
                    "seller_feedback_pct": listing.get("seller_feedback_pct"),
                    "seller_feedback_score": listing.get("seller_feedback_score"),
                    "buying_options": listing.get("buying_options"),
                    "shipping_cost": listing.get("shipping_cost"),
                    "shipping_currency": listing.get("shipping_currency"),
                    "shipping_cost_type": listing.get("shipping_cost_type"),
                    "min_delivery_date": listing.get("min_delivery_date"),
                    "max_delivery_date": listing.get("max_delivery_date"),
                    "brand": monitor.get("brand"),
                    "model": monitor.get("model"),
                    "color": monitor.get("color"),
                    "screen_size_in": monitor.get("screen_size_in"),
                    "resolution_width": (monitor.get("resolution") or {}).get("width"),
                    "resolution_height": (monitor.get("resolution") or {}).get("height"),
                    "refresh_rate_hz": monitor.get("refresh_rate_hz"),
                    "panel_type": monitor.get("panel_type"),
                    "aspect_ratio": monitor.get("aspect_ratio"),
                    "response_time_ms": monitor.get("response_time_ms"),
                    "brightness_nits": monitor.get("brightness_nits"),
                    "hdr": monitor.get("hdr"),
                    "has_adaptive_sync": monitor.get("has_adaptive_sync"),
                    "ports": self._ports_to_string(monitor.get("ports", [])),
                    "width_in": (monitor.get("dimensions_in") or {}).get("width"),
                    "height_in": (monitor.get("dimensions_in") or {}).get("height"),
                    "depth_in": (monitor.get("dimensions_in") or {}).get("depth"),
                }
            )
        return combined

    def _ports_to_string(self, ports):
        if not ports:
            return None
        parts = []
        for port in ports:
            port_type = port.get("type")
            count = port.get("count")
            version = port.get("version")
            label = f"{count}x {port_type}" if count is not None else f"{port_type}"
            if version:
                label = f"{label} {version}"
            parts.append(label)
        return "; ".join(parts) if parts else None

    def _normalize_title(self, title):
        if not title:
            return None
        title = title.lower()
        title = re.sub(r"[^a-z0-9]+", " ", title)
        return re.sub(r"\\s+", " ", title).strip()

    def _non_null_score(self, monitor):
        keys = [
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
        ]
        score = 0
        for key in keys:
            value = monitor.get(key)
            if key == "resolution":
                value = (value or {}).get("label")
            if value is not None:
                score += 1
        return score

    def _build_best_index(self, monitors, key):
        index = {}
        for monitor in monitors:
            if key == "_title_key":
                monitor["_title_key"] = self._normalize_title(monitor.get("title"))
            value = monitor.get(key)
            if not value:
                continue
            if value not in index or self._non_null_score(monitor) > self._non_null_score(index[value]):
                index[value] = monitor
        return index


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
