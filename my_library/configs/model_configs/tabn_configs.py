from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.utils.env_loader import use_gpu_enabled


@dataclass
class TabNetConfig(ConfigBase):
    """Configuration for TabNet-based deep learning models.

    Attributes:
        model_name (str): Identifier for the TabNet model.
        task_type (Literal): Prediction task type.
        use_gpu (bool): Whether to use GPU.
        save_log (bool): Whether to save logs.
        params (Dict): TabNet-specific hyperparameters.
    """
    model_name: str = "tabnet"
    task_type: Literal["classification", "regression"] = "regression"
    use_gpu: bool = use_gpu_enabled()
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-3,
        "momentum": 0.3,
        "clip_value": 2.0,
        "batch_size": 32,
        "virtual_batch_size": 16,
        "lr": 2e-2,
        "max_epochs": 200,
        "patience": 30,
        "early_stopping": True,
        "verbose": 0,
        "seed": 42
    })
