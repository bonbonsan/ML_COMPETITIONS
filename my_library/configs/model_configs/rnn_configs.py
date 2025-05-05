from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class RNNConfig(ConfigBase):
    """Configuration for RNN-based deep learning models.

    Attributes:
        model_name (str): Identifier for the RNN model.
        task_type (Literal): Prediction task type.
        use_gpu (bool): Whether to use GPU.
        save_log (bool): Whether to save logs.
        params (Dict): RNN-specific hyperparameters.
    """
    model_name: str = "rnn"
    task_type: Literal["classification", "regression"] = "regression"
    use_gpu: bool = False
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "input_size": 1,
        "output_size": 1,
        "hidden_size": 64,
        "num_layers": 1,
        "activation": "relu",
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 300,
        "batch_size": 32
    })
