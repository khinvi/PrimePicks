"""Download, stream, and filter the Amazon Reviews 2023 dataset."""

import csv
import gzip
import json
import os
import re

import requests
from tqdm import tqdm

import config


def download_file(url: str, dest_path: str) -> None:
    """Stream-download a file with progress bar and resume support."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check for partial download to support resume
    downloaded = 0
    headers = {}
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)
        headers["Range"] = f"bytes={downloaded}-"

    resp = requests.get(url, stream=True, headers=headers, timeout=30)

    # If server doesn't support range requests, start over
    if resp.status_code == 200:
        downloaded = 0
        mode = "wb"
    elif resp.status_code == 206:
        mode = "ab"
    else:
        resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + downloaded

    with (
        open(dest_path, mode) as f,
        tqdm(total=total, initial=downloaded, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=config.DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))


def stream_jsonl_gz(filepath: str):
    """Yield parsed JSON objects from a gzipped JSONL file, one line at a time."""
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def filter_athletic_footwear_asins(meta_filepath: str) -> set[str]:
    """
    Stream through metadata and identify athletic footwear parent ASINs.

    Strategy:
    1. Check main_category or categories for athletic/shoe categories
    2. Check title for brand/model keywords
    3. Require footwear-related terms to exclude apparel/accessories

    Returns a set of parent_asin strings.
    """
    valid_asins = set()
    item_metadata = {}

    print("Filtering athletic footwear ASINs from metadata...")
    for item in tqdm(stream_jsonl_gz(meta_filepath), desc="Scanning metadata"):
        title = item.get("title", "")
        parent_asin = item.get("parent_asin", "")
        if not parent_asin or not title:
            continue

        # Build a searchable text blob from relevant fields
        main_cat = item.get("main_category", "")
        categories = " ".join(item.get("categories", []))  # flat list of strings
        description = " ".join(item.get("description", []))
        features = " ".join(item.get("features", []))
        search_text = f"{title} {main_cat} {categories} {description} {features}"

        # Must be footwear (not clothing, accessories, etc.)
        is_footwear = _matches_keywords(search_text, config.FOOTWEAR_TERMS)
        if not is_footwear:
            continue

        # Must match athletic category OR brand keywords
        is_athletic = (
            _matches_keywords(search_text, config.CATEGORY_KEYWORDS)
            or _matches_keywords(title, config.BRAND_KEYWORDS)
        )
        if not is_athletic:
            continue

        valid_asins.add(parent_asin)
        item_metadata[parent_asin] = {
            "title": title,
            "main_category": main_cat,
            "average_rating": item.get("average_rating"),
            "rating_number": item.get("rating_number"),
            "price": item.get("price"),
            "images": (item.get("images", [{}])[0].get("large") if item.get("images") else None),
        }

    print(f"Found {len(valid_asins)} athletic footwear items.")

    # Save results
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(config.ATHLETIC_ASINS_PATH, "w") as f:
        json.dump(list(valid_asins), f)
    with open(config.ITEM_METADATA_PATH, "w") as f:
        json.dump(item_metadata, f)

    return valid_asins


def load_athletic_asins() -> set[str]:
    """Load previously filtered ASIN set from disk."""
    with open(config.ATHLETIC_ASINS_PATH) as f:
        return set(json.load(f))


def extract_review_triplets(review_filepath: str, valid_asins: set[str]) -> str:
    """
    Stream reviews and extract (user_id, parent_asin, rating) for valid items.
    Writes results to a CSV file and returns the output path.
    """
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    output_path = config.TRIPLETS_PATH
    count = 0

    print("Extracting review triplets...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "parent_asin", "rating"])

        for review in tqdm(stream_jsonl_gz(review_filepath), desc="Scanning reviews"):
            parent_asin = review.get("parent_asin", "")
            if parent_asin not in valid_asins:
                continue

            user_id = review.get("user_id", "")
            rating = review.get("rating")
            if not user_id or rating is None:
                continue

            writer.writerow([user_id, parent_asin, rating])
            count += 1

    print(f"Extracted {count} review triplets to {output_path}")
    return output_path


def load_item_metadata() -> dict:
    """Load item metadata from disk."""
    with open(config.ITEM_METADATA_PATH) as f:
        return json.load(f)
