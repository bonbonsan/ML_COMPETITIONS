from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class HistGBDTConfig(ConfigBase):
    """Configuration class for sklearn's HistGradientBoosting models.

    This configuration is used to initialize a CustomHistGBDT model
    with parameters suitable for either classification or regression tasks.
    GPU acceleration is not supported.

    Attributes:
        model_name (str): Name used to identify the model.
        task_type (Literal): 'classification' or 'regression'.
        use_gpu (bool): Always False; GPU is not supported.
        save_log (bool): Whether to save logs during training.
        params (Dict): Dictionary of model hyperparameters.
    """
    model_name: str = "histgbdt"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = False  # Should always remain False
    save_log: bool = True

    params: Dict = field(default_factory=lambda: {
        # core parameters
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": None,
        "random_state": 42,
        "l2_regularization": 0.0,

        # --- early stopping parameters for sklearn HistGBDT ---
        "early_stopping": False,         # enable early stopping
        "n_iter_no_change": 10,         # no improvement rounds
        "validation_fraction": 0.1,     # train データの何割を内部検証に使うか
        "tol": 1e-4,                    # 最小改善量の閾値
        # scoring を指定すればデフォルトの「loss」以外も使える
        "scoring": None,

        # verbosity
        "verbose": 0,
    })
