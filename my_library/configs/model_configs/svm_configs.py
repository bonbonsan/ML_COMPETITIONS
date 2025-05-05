from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class SVMConfig(ConfigBase):
    """Configuration class for sklearn-based SVM models.

    Attributes:
        model_name (str): Model identifier.
        task_type (Literal['classification','regression']): Task type.
        use_gpu (bool): Whether to use GPU (not supported by sklearn SVM).
        save_log (bool): Whether to save training logs.
        params (Dict): Hyperparameters for SVC/SVR.
    """
    model_name: str = "svm"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = False
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    })
