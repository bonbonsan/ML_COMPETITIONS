from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.utils.env_loader import use_gpu_enabled


@dataclass
class XGBoostConfig(ConfigBase):
    """Configuration class for XGBoost models.

    This class defines the configuration for XGBoost classifiers or regressors,
    inheriting the standard structure from GBDTConfigBase.

    Attributes:
        model_name (str): Model identifier (default: 'xgboost').
        task_type (Literal): Task type - 'classification' or 'regression'.
        use_gpu (bool): Whether to use GPU acceleration.
        save_log (bool): Whether to save training logs.
        params (Dict): Hyperparameters specific to XGBoost.
    """
    model_name: str = "xgboost"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = use_gpu_enabled()
    save_log: bool = True
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "auto",
        "early_stopping_rounds": None,  # 10
        # "eval_metric": "logloss",
    })
