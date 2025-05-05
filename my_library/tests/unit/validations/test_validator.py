import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, root_mean_squared_error

from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.custom_cbt import CustomCatBoost
from my_library.utils.data_loader import load_sample_data
from my_library.validations.validator import Validator


# pytest my_library/tests/unit/validations/test_validator.py -v
@pytest.mark.parametrize("task_type,dataset_name,target_column", [
    ("classification", "iris", "target"),
    ("regression", "diabetes", "target"),
])
def test_validator_with_catboost(task_type, dataset_name, target_column):
    # Load sample data for classification or regression
    df = load_sample_data(name=dataset_name, task=task_type)
    X_train = df.drop(columns=target_column)
    y_train = df[target_column]

    # Define fit configuration
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

    # Initialize CatBoost model and validator
    config = CatBoostConfig(task_type=task_type, use_gpu=False, save_log=False)
    model = CustomCatBoost(config=config)
    validator = Validator(model)

    # Train
    validator.train(X_train, y_train, fit_config)

    # Predict
    preds = validator.predict(X_train)
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == X_train.shape[0]
    assert preds.name == y_train.name or preds.name == target_column

    # Evaluate default metric
    score = validator.evaluate(y_train, preds)
    assert isinstance(score, float)
    if task_type == "classification":
        # Accuracy should be between 0 and 1
        assert 0.0 <= score <= 1.0
        # Compare to sklearn accuracy
        assert score == pytest.approx(accuracy_score(y_train, preds))
    else:
        # RMSE should be non-negative
        assert score >= 0.0
        # Compare to sklearn RMSE
        rmse = root_mean_squared_error(y_train, preds)
        assert score == pytest.approx(rmse)

    # Evaluate with custom metric
    def custom_metric(y_true, y_pred):
        return 3.1415
    assert validator.evaluate(y_train, preds, metric_fn=custom_metric) == pytest.approx(3.1415)

    # Ensure validator does not alter original model behavior
    preds2 = model.predict(X_train)
    assert preds.equals(preds2)
