from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class RandomForestConfig(ConfigBase):
    """Configuration class for sklearn-based random forest models.

    Attributes:
        model_name (str): Identifier for the model instance.
        task_type (Literal): 'classification' or 'regression'.
        use_gpu (bool): Whether to use GPU acceleration (not supported in sklearn).
        save_log (bool): Whether to persist training logs to file.
        params (Dict): Hyperparameters passed to the sklearn RandomForest constructor.
    """
    model_name: str = "random_forest"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = False
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        # you can add other RF-specific params here
    })
