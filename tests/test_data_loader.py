"""Tests for the data loading and filtering module."""

import csv
import gzip
import json
import os
import tempfile

import pytest

from src.data_loader import (
    _matches_keywords,
    extract_review_triplets,
    filter_athletic_footwear_asins,
    stream_jsonl_gz,
)
import config


@pytest.fixture
def tmp_dir(tmp_path):
    """Override config paths to use temp directory."""
    original_processed = config.PROCESSED_DIR
    original_asins = config.ATHLETIC_ASINS_PATH
    original_metadata = config.ITEM_METADATA_PATH

    config.PROCESSED_DIR = str(tmp_path / "processed")
    config.ATHLETIC_ASINS_PATH = str(tmp_path / "processed" / "athletic_asins.json")
    config.ITEM_METADATA_PATH = str(tmp_path / "processed" / "item_metadata.json")

    yield tmp_path

    config.PROCESSED_DIR = original_processed
    config.ATHLETIC_ASINS_PATH = original_asins
    config.ITEM_METADATA_PATH = original_metadata


def _create_jsonl_gz(filepath, records):
    """Helper to create a gzipped JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestStreamJsonlGz:
    def test_reads_all_records(self, tmp_path):
        filepath = str(tmp_path / "test.jsonl.gz")
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        _create_jsonl_gz(filepath, records)

        result = list(stream_jsonl_gz(filepath))
        assert result == records

    def test_skips_blank_lines(self, tmp_path):
        filepath = str(tmp_path / "test.jsonl.gz")
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')

        result = list(stream_jsonl_gz(filepath))
        assert len(result) == 2


class TestMatchesKeywords:
    def test_match_found(self):
        assert _matches_keywords("Nike Air Max Running Shoes", ["nike", "running"])

    def test_case_insensitive(self):
        assert _matches_keywords("NIKE AIR MAX", ["nike"])

    def test_no_match(self):
        assert not _matches_keywords("Cotton T-Shirt", ["nike", "shoe"])


class TestFilterAthleticFootwearAsins:
    def test_filters_correctly(self, tmp_dir):
        meta_path = str(tmp_dir / "meta.jsonl.gz")
        records = [
            {
                "parent_asin": "SHOE001",
                "title": "Nike Air Max Running Shoes",
                "main_category": "Clothing, Shoes & Jewelry",
                "categories": ["Athletic"],
                "description": ["Great running shoe"],
                "features": [],
            },
            {
                "parent_asin": "SHIRT001",
                "title": "Cotton T-Shirt",
                "main_category": "Clothing",
                "categories": ["Tops"],
                "description": ["Comfy shirt"],
                "features": [],
            },
            {
                "parent_asin": "SHOE002",
                "title": "Adidas Ultraboost Sneaker",
                "main_category": "Clothing, Shoes & Jewelry",
                "categories": ["Shoes"],
                "description": ["Athletic sneaker for training"],
                "features": [],
            },
        ]
        _create_jsonl_gz(meta_path, records)

        result = filter_athletic_footwear_asins(meta_path)

        assert "SHOE001" in result
        assert "SHOE002" in result
        assert "SHIRT001" not in result


class TestExtractReviewTriplets:
    def test_extracts_valid_reviews(self, tmp_dir):
        reviews_path = str(tmp_dir / "reviews.jsonl.gz")
        records = [
            {"user_id": "U1", "parent_asin": "SHOE001", "rating": 5.0, "asin": "V1"},
            {"user_id": "U2", "parent_asin": "INVALID", "rating": 3.0, "asin": "V2"},
            {"user_id": "U3", "parent_asin": "SHOE001", "rating": 4.0, "asin": "V3"},
        ]
        _create_jsonl_gz(reviews_path, records)

        valid_asins = {"SHOE001"}

        config.TRIPLETS_PATH = str(tmp_dir / "processed" / "triplets.csv")
        os.makedirs(os.path.dirname(config.TRIPLETS_PATH), exist_ok=True)

        output = extract_review_triplets(reviews_path, valid_asins)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["user_id"] == "U1"
        assert rows[0]["parent_asin"] == "SHOE001"
        assert float(rows[0]["rating"]) == 5.0
