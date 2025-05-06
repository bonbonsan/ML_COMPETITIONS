from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.utils.env_loader import use_gpu_enabled


@dataclass
class LightGBMConfig(ConfigBase):
    """Configuration class for LightGBM models.

    This class defines the configuration for LightGBM classifiers or regressors,
    inheriting the standard structure from GBDTConfigBase.

    Attributes:
        model_name (str): Model identifier (default: 'lightgbm').
        task_type (Literal): Task type - 'classification' or 'regression'.
        use_gpu (bool): Whether to use GPU acceleration.
        save_log (bool): Whether to save training logs.
        params (Dict): Hyperparameters specific to LightGBM.
    """
    model_name: str = "lightgbm"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = use_gpu_enabled()
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1
    })
