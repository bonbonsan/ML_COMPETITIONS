# models

Model wrappers that unify third-party libraries under `CustomModelInterface`. Every implementation exposes `build_model`, `fit`, `predict`, `predict_proba`, feature importance helpers, and persistence utilities.

## Key Files

- `interface.py` – Abstract base class defining the contract for all custom models.
- `factory.py` – Central factory that constructs model instances based on `ModelType` and a matching config dataclass.
- `custom_*.py` – Concrete wrappers for LightGBM, XGBoost, CatBoost, HistGBDT, linear models, SVM, KNN, Random Forest, TabNet, and neural architectures (RNN, LSTM, Transformer).

## Usage Pattern

```python
from my_library.models.factory import CustomModelFactory, ModelType
from my_library.configs.model_configs.lightgbm_configs import LightGBMConfig

config = LightGBMConfig(task_type="classification", params={"n_estimators": 500})
model = CustomModelFactory.create_model(ModelType.LIGHTGBM, config=config)
model.fit(X_train, y_train, fit_config)
preds = model.predict(X_valid)
```

When adding a new algorithm, implement the interface, create a config dataclass, register it inside `CustomModelFactory`, and add unit tests under `my_library/tests/unit/models`.
