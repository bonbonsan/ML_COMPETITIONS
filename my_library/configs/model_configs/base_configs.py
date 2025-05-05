from abc import ABC
from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class ConfigBase(ABC):
    """Abstract base class for model configuration.

    This class defines the common structure for all model configurations,
    including model identity, task type, hardware usage, and hyperparameters.

    Attributes:
        model_name (str): Name identifier for the model.
        task_type (Literal): Either 'classification' or 'regression'.
        use_gpu (bool): Indicates whether to use GPU acceleration.
        save_log (bool): Whether to save logs during training.
        params (Dict): Hyperparameter dictionary for the model.
    """
    model_name: str
    task_type: Literal["classification", "regression"]
    use_gpu: bool
    save_log: bool
    params: Dict
