# configs

Dataclass configuration objects that define runtime behavior for every model and training routine. Centralizing configuration keeps experiments reproducible and makes it easy to swap models inside validation pipelines.

## Key Modules

- `model_configs/base_configs.py` – Abstract base dataclass that captures shared fields such as `model_name`, `task_type`, `use_gpu`, and `params`.
- `model_configs/*_configs.py` – Concrete configs for individual models (LightGBM, XGBoost, CatBoost, neural nets, sklearn estimators, TabNet, etc.).
- `model_configs/fit_configs.py` – Training-time settings (feature subset, evaluation sets, early stopping rounds, epochs, batch size).

## Usage

```python
from my_library.configs.model_configs.lightgbm_configs import LightGBMConfig

config = LightGBMConfig(task_type="regression", params={"learning_rate": 0.03})
model = CustomModelFactory.create_model(ModelType.LIGHTGBM, config=config)
```

When extending the library with a new model, create a matching config dataclass here and register it with `CustomModelFactory`. Keep defaults conservative so unit tests pass on CPU-only environments.
