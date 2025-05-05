from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class LinearConfig(ConfigBase):
    """Configuration class for sklearn-based linear models.

    Attributes:
        model_name (str): Model identifier (default: 'sklearn_linear').
        task_type (Literal): Task type - 'classification' or 'regression'.
        use_gpu (bool): Whether to use GPU acceleration.
        save_log (bool): Whether to save training logs.
        params (Dict): Hyperparameters for the sklearn model.
    """
    model_name: str = "linear"
    task_type: Literal["classification", "regression"] = "regression"
    use_gpu: bool = False
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        # Common linear model parameters
        "fit_intercept": True,
        # For LogisticRegression, solver defaults to 'lbfgs'
    })
