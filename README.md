# PrimePicks

An athletic sneaker recommendation engine powered by item-item collaborative filtering, built on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset.

## How It Works

PrimePicks filters the Amazon Clothing, Shoes & Jewelry dataset down to athletic footwear (basketball, running, training shoes from Nike, Adidas, Jordan, etc.), then builds an item-item collaborative filtering model:

1. **Data Pipeline** - Streams multi-GB JSONL files, filters metadata to identify athletic footwear ASINs, extracts `(user_id, parent_asin, rating)` triplets
2. **Preprocessing** - Maps string IDs to integers, builds a sparse user-item interaction matrix (`scipy.sparse.csr_matrix`)
3. **Training** - Computes pairwise cosine similarity between all items, truncates to top-k neighbors
4. **Prediction** - Predicts ratings via weighted average: `r_hat(u,i) = sum(sim(i,j) * r(u,j)) / sum(|sim(i,j)|)`
5. **API** - Serves recommendations through a FastAPI REST API

## Project Structure

```
PrimePicks/
├── config.py                # URLs, paths, filtering keywords, hyperparameters
├── src/
│   ├── data_loader.py       # Download, stream, filter datasets
│   ├── preprocessing.py     # ID translation, sparse matrix construction
│   ├── train.py             # Item-item cosine similarity computation
│   ├── predict.py           # Rating prediction & top-N recommendations
│   └── evaluate.py          # RMSE, MAE evaluation
├── api/
│   ├── main.py              # FastAPI app with lifespan model loader
│   ├── routes.py            # API endpoints
│   └── schemas.py           # Pydantic request/response models
├── scripts/
│   ├── download_data.py     # Download raw data
│   └── run_pipeline.py      # End-to-end pipeline
└── tests/                   # Unit tests (36 tests)
```

## Setup

```bash
# Clone and install
git clone https://github.com/khinvi/PrimePicks.git
cd PrimePicks
pip install -r requirements.txt

# Run the full pipeline (download, filter, preprocess, train, evaluate)
python scripts/run_pipeline.py

# Start the API server
uvicorn api.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model stats (users, items, sparsity) |
| `POST` | `/recommend` | Top-N recommendations for a user |
| `POST` | `/predict` | Predict rating for a user-item pair |
| `GET` | `/similar/{asin}?k=10` | Most similar items to a given ASIN |

### Example: Get Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "AEYORY2AVPMCPDV57CE337YU5LXA", "n": 5}'
```

### Example: Find Similar Items

```bash
curl http://localhost:8000/similar/B088SZDGXG?k=5
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error - penalizes large prediction errors |
| **MAE** | Mean Absolute Error - average prediction error magnitude |
| **Coverage** | Fraction of test ratings the model can predict |

## Testing

```bash
pytest tests/ -v
```

## Technical Details

- **Model**: Item-item collaborative filtering with cosine similarity
- **Matrix**: Sparse CSR format via `scipy.sparse` (handles high sparsity efficiently)
- **Filtering**: Uses `parent_asin` to aggregate size/color variants into one item
- **Cold start**: Unknown users receive popularity-based recommendations
- **Neighbors**: Top-k truncation (default k=50) to reduce noise
