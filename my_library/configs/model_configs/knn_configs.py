from dataclasses import dataclass, field
from typing import Dict, Literal

from my_library.configs.model_configs.base_configs import ConfigBase


@dataclass
class KNNConfig(ConfigBase):
    """Configuration for sklearn-based kNN models.

    Attributes:
        model_name (str): Identifier for the kNN model (default: 'knn').
        task_type (Literal['classification', 'regression']): Specifies whether to use
            KNeighborsClassifier or KNeighborsRegressor.
        use_gpu (bool): GPU support flag (ignored, as kNN does not use GPU).
        save_log (bool): Whether to save training logs to file.
        params (Dict): Hyperparameters for the sklearn kNN model:
            - n_neighbors (int): Number of neighbors to use.
            - weights (str or callable): Weight function.
            - algorithm (str): Algorithm to compute nearest neighbors.
            - leaf_size (int): Leaf size for tree-based algorithms.
            - p (int): Power parameter for the Minkowski metric.
            - metric (str): Distance metric.
    """
    model_name: str = "knn"
    task_type: Literal["classification", "regression"] = "classification"
    use_gpu: bool = False
    save_log: bool = False
    params: Dict = field(default_factory=lambda: {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski"
    })
