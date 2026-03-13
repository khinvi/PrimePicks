#!/usr/bin/env python3
"""Download the raw Amazon Reviews 2023 dataset files."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import download_file


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)

    print("=== Downloading Metadata ===")
    if os.path.exists(config.RAW_METADATA_PATH):
        print(f"Metadata already exists at {config.RAW_METADATA_PATH}, skipping.")
    else:
        download_file(config.METADATA_URL, config.RAW_METADATA_PATH)

    print("\n=== Downloading Reviews ===")
    if os.path.exists(config.RAW_REVIEWS_PATH):
        print(f"Reviews already exist at {config.RAW_REVIEWS_PATH}, skipping.")
    else:
        download_file(config.REVIEWS_URL, config.RAW_REVIEWS_PATH)

    print("\nDone! Raw data saved to:", config.RAW_DIR)


if __name__ == "__main__":
    main()
