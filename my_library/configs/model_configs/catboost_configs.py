from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class CatBoostConfig(ConfigBase):
    """Configuration class for CatBoost models.

    This class defines the configuration for CatBoost classifiers or regressors,
    inheriting the standard structure from GBDTConfigBase.

    Attributes:
        model_name (str): Model identifier (default: 'catboost').
        task_type (Literal): Task type - 'classification' or 'regression'.
        use_gpu (bool): Whether to use GPU acceleration.
        save_log (bool): Whether to save training logs.
        params (Dict): Hyperparameters specific to CatBoost.
    """
    model_name: str = "catboost"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = False
    save_log: bool = True
    params: Dict = field(default_factory=lambda: {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": 0,
        # "task_type": "CPU"
    })