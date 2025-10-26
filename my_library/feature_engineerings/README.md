# feature_engineerings

Feature engineering blocks that can be reused across competitions. Modules range from simple aggregations to advanced embeddings and time-aware transforms.

## Highlights

- `pipeline.py` – Lightweight orchestration helper to chain feature builders.
- `categorical_aggregations.py`, `rolling_stats.py`, `statistics.py` – Aggregation utilities for tabular datasets.
- `encoders.py` – Target and frequency encoding variants.
- `word2vec.py`, `itemcf.py`, `signal_features.py` – Representation learning for sequential, recommender, or signal data.
- `graph_prone_features.py`, `prone.py` – Graph-based embeddings (ProNE) and related helpers.
- `temporals.py` – Time series friendly transformers (lags, windows, calendar features).
- `dae.py` – Denoising autoencoder based feature augmentation.

## Usage

Import the desired transformer and apply it inside your preprocessing workflow:

```python
from my_library.feature_engineerings.statistics import build_numeric_stats

features = build_numeric_stats(df, group_keys=["customer_id"], cols=["amount"]) 
```

Keep functions side-effect free when possible so they compose cleanly in pipelines and tests remain deterministic.
