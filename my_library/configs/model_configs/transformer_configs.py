from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.utils.env_loader import use_gpu_enabled


@dataclass
class TransformerConfig(ConfigBase):
    """Configuration for Transformer-based deep learning models.

    Attributes:
        model_name (str): Identifier for the Transformer model.
        task_type (Literal): Prediction task type.
        use_gpu (bool): Whether to use GPU.
        save_log (bool): Whether to save logs.
        params (Dict): Transformer model parameters.
    """
    model_name: str = "transformer"
    task_type: Literal["regression", "classification"] = "regression"
    use_gpu: bool = use_gpu_enabled()
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "input_size": 1,
        "output_size": 1,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "lr": 0.001,
        "activation": "relu",
        "epochs": 100,
        "batch_size": 32
    })
