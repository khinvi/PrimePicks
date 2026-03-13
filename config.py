"""Central configuration for the PrimePicks recommendation engine."""

import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# =============================================================================
# Amazon Reviews 2023 Dataset URLs (Clothing, Shoes & Jewelry)
# https://amazon-reviews-2023.github.io/
# =============================================================================
REVIEWS_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
    "Clothing_Shoes_and_Jewelry.jsonl.gz"
)
METADATA_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"
    "meta_Clothing_Shoes_and_Jewelry.jsonl.gz"
)

# Raw file paths
RAW_REVIEWS_PATH = os.path.join(RAW_DIR, "Clothing_Shoes_and_Jewelry.jsonl.gz")
RAW_METADATA_PATH = os.path.join(RAW_DIR, "meta_Clothing_Shoes_and_Jewelry.jsonl.gz")

# Processed file paths
ATHLETIC_ASINS_PATH = os.path.join(PROCESSED_DIR, "athletic_asins.json")
TRIPLETS_PATH = os.path.join(PROCESSED_DIR, "triplets.csv")
TRAIN_TRIPLETS_PATH = os.path.join(PROCESSED_DIR, "train_triplets.csv")
TEST_TRIPLETS_PATH = os.path.join(PROCESSED_DIR, "test_triplets.csv")
USER_MAPPING_PATH = os.path.join(PROCESSED_DIR, "user_to_idx.json")
ITEM_MAPPING_PATH = os.path.join(PROCESSED_DIR, "item_to_idx.json")
INTERACTION_MATRIX_PATH = os.path.join(PROCESSED_DIR, "interaction_matrix.npz")
SIMILARITY_MATRIX_PATH = os.path.join(PROCESSED_DIR, "similarity_matrix.npz")
ITEM_METADATA_PATH = os.path.join(PROCESSED_DIR, "item_metadata.json")

# =============================================================================
# Athletic Footwear Filtering
# =============================================================================
# Category keywords (matched against category lists and main_category)
CATEGORY_KEYWORDS = [
    "athletic", "basketball", "running", "training", "sneaker",
    "walking", "tennis", "cross training", "trail running",
]

# Brand/model keywords (matched against title)
BRAND_KEYWORDS = [
    "nike", "adidas", "jordan", "new balance", "asics", "brooks",
    "puma", "reebok", "under armour", "saucony", "hoka", "mizuno",
    "kobe", "g.t. cut", "lebron", "kyrie", "air max", "air force",
    "ultraboost", "gel-kayano", "fresh foam", "pegasus",
]

# Terms that must appear to confirm it's footwear (not apparel/accessories)
FOOTWEAR_TERMS = [
    "shoe", "sneaker", "footwear", "trainer", "boot", "cleat",
]

# =============================================================================
# Preprocessing Hyperparameters
# =============================================================================
MIN_RATINGS_PER_USER = 3
MIN_RATINGS_PER_ITEM = 5
TEST_RATIO = 0.2

# =============================================================================
# Model Hyperparameters
# =============================================================================
K_NEIGHBORS = 50          # Number of similar items to keep per item
DEFAULT_TOP_N = 10        # Default number of recommendations to return
CO_PURCHASE_BOOST = 0.1   # Additive similarity boost for co-purchased items

# =============================================================================
# Download Settings
# =============================================================================
DOWNLOAD_CHUNK_SIZE = 8192
