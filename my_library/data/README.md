# data

Sample datasets used for unit tests, end-to-end examples, and quick sanity checks. Files are intentionally small to keep repository size manageable.

## Structure

- `samples/classification/` – CSVs for classification tasks (`iris`, `cancer`, `titanic`).
- `samples/regression/` – CSVs for regression tasks (`diabetes`, etc.).

## Loading Helper

Use `my_library.utils.data_loader.load_sample_data` to access these datasets so downstream code does not need to manage paths.

```python
from my_library.utils.data_loader import load_sample_data

iris = load_sample_data(name="iris", task="classification")
```

Large or private datasets should live outside the repository and be mounted into the Docker container as needed.
