from dotenv import load_dotenv

import sys
import os
import requests

from src.exception import CustomException
from src.logger import logging
from src.utils import save_raw_json

load_dotenv()

EBAY_OAUTH_TOKEN = os.getenv("EBAY_OAUTH_TOKEN")
BASE_URL = "https://api.ebay.com"
SEARCH_ENDPOINT = "/buy/browse/v1/item_summary/search"


def search_ebay_products(
    query: str,
    category_id: str,
    limit: int = 200,
    offset: int = 0
):
    """
    Search active eBay listings using the Browse API.
    Returns active marketplace data (not sold status).
    """
    try:
        if not EBAY_OAUTH_TOKEN:
            raise ValueError("EBAY_OAUTH_TOKEN not found in environment variables.")

        headers = {
            "Authorization": f"Bearer {EBAY_OAUTH_TOKEN}",
            "Content-Type": "application/json",
        }

        params = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "fieldgroups": "EXTENDED",
        }

        if category_id:
            params["category_ids"] = category_id

        logging.info(f"Calling eBay Browse API for query: {query}")

        response = requests.get(
            url=BASE_URL + SEARCH_ENDPOINT,
            headers=headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        return data

    except Exception as e:
        raise CustomException(e, sys)


def fetch_all_listings(query: str, category_id: str, max_results: int | None = None):
    all_items = []
    offset = 0
    limit = 200
    while True:
        data = search_ebay_products(query=query, category_id=category_id, limit=limit, offset=offset)
        items = data.get("itemSummaries", [])
        all_items.extend(items)
        total = data.get("total", len(all_items))
        offset += len(items)
        if not items or offset >= total:
            break
        if max_results is not None and len(all_items) >= max_results:
            all_items = all_items[:max_results]
            break
    return {"itemSummaries": all_items, "total": len(all_items), "query": query, "category_id": category_id}


if __name__ == "__main__":
    # --- CONFIG: edit these values for each run ---
    query = "budget monitor"
    category_id = "58058"  # Monitors
    max_results = None  # set to an int to cap results, or leave None for all
    # ---------------------------------------------

    ebay_data = fetch_all_listings(
        query=query,
        category_id=category_id,
        max_results=max_results,
    )

    query_slug = query.replace(" ", "_").lower()
    save_raw_json(
        ebay_data,
        f"notebook/data/ebay/{query_slug}/listings"
    )
