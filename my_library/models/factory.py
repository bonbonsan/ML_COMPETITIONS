from enum import Enum, auto
from typing import Dict, Optional, Tuple, Type

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.histgbdt_configs import HistGBDTConfig
from my_library.configs.model_configs.knn_configs import KNNConfig
from my_library.configs.model_configs.lightgbm_configs import LightGBMConfig
from my_library.configs.model_configs.linear_configs import LinearConfig
from my_library.configs.model_configs.lstm_configs import LSTMConfig
from my_library.configs.model_configs.rf_configs import RandomForestConfig
from my_library.configs.model_configs.rnn_configs import RNNConfig
from my_library.configs.model_configs.svm_configs import SVMConfig
from my_library.configs.model_configs.tabn_configs import TabNetConfig
from my_library.configs.model_configs.transformer_configs import TransformerConfig
from my_library.configs.model_configs.xgboost_configs import XGBoostConfig
from my_library.models.custom_cbt import CustomCatBoost
from my_library.models.custom_histgbdt import CustomHistGBDT
from my_library.models.custom_knn import CustomKNN
from my_library.models.custom_lgm import CustomLightGBM
from my_library.models.custom_linear import CustomLinear
from my_library.models.custom_lstm import CustomLSTMModel
from my_library.models.custom_rf import CustomRandomForest
from my_library.models.custom_rnn import CustomRNNModel
from my_library.models.custom_svm import CustomSVM
from my_library.models.custom_tabn import CustomTabNetModel
from my_library.models.custom_transformer import CustomTransformerModel
from my_library.models.custom_xgb import CustomXGBoost
from my_library.models.interface import CustomModelInterface


class ModelType(Enum):
    """Enumeration for supported machine learning model types."""
    # GBDT
    LIGHTGBM = auto()
    XGBOOST = auto()
    CATBOOST = auto()
    HISTGBDT = auto()

    # Deep Learning
    LSTM = auto()
    RNN = auto()
    TRANSFORMER = auto()
    TABNET = auto()

    # sklearn
    LINEAR = auto()
    KNN = auto()
    SVM = auto()
    RF = auto()


class CustomModelFactory:
    """Factory class for building custom machine learning model instances.

    This class provides a mechanism to build specific model instances based on
    a given ModelType. If no configuration is provided, a default configuration
    associated with the model type is used.

    Class Attributes:
        model_map (Dict[ModelType, Tuple[Type[CustomModelInterface], Type[ConfigBase]]]):
            Mapping of ModelType to corresponding model class and config class.
    """

    model_map: Dict[ModelType, Tuple[Type[CustomModelInterface], Type[ConfigBase]]] = {
        ModelType.LIGHTGBM: (CustomLightGBM, LightGBMConfig),
        ModelType.XGBOOST: (CustomXGBoost, XGBoostConfig),
        ModelType.CATBOOST: (CustomCatBoost, CatBoostConfig),
        ModelType.HISTGBDT: (CustomHistGBDT, HistGBDTConfig),
        ModelType.RNN: (CustomRNNModel, RNNConfig),
        ModelType.LSTM: (CustomLSTMModel, LSTMConfig),
        ModelType.TRANSFORMER: (CustomTransformerModel, TransformerConfig),
        ModelType.TABNET: (CustomTabNetModel, TabNetConfig),
        ModelType.LINEAR: (CustomLinear, LinearConfig),
        ModelType.KNN: (CustomKNN, KNNConfig),
        ModelType.SVM: (CustomSVM, SVMConfig),
        ModelType.RF: (CustomRandomForest, RandomForestConfig),
    }

    @classmethod
    def create_model(cls, model_type: ModelType,
                     config: Optional[ConfigBase] = None) -> CustomModelInterface:
        """Build and return an instance of a custom model.

        Args:
            model_type (ModelType): The type of model to build.
            config (Optional[ConfigBase]): An optional configuration instance. If None,
                a default configuration for the specified model_type is used.

        Returns:
            CustomModelInterface: An instance of the requested custom model.

        Raises:
            ValueError: If the model_type is not supported.
            TypeError: If the provided config is not an instance of the expected config class.
        """
        if model_type not in cls.model_map:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class, config_class = cls.model_map[model_type]

        if config is None:
            # Use default config for the model type.
            config = config_class()

        if not isinstance(config, config_class):
            raise TypeError(f"""Expected config of type {config_class.__name__},
                            got {type(config).__name__}""")

        model_instance = model_class(config)
        return model_instance


if __name__ == "__main__":
    # Example usage of CustomModelFactory with default LightGBM config
    lgm_model = CustomModelFactory.create_model(ModelType.LIGHTGBM)
    print(lgm_model)

    # Example usage with custom Transformer config
    custom_config = TransformerConfig(task_type="classification", use_gpu=True)
    transformer_model = CustomModelFactory.create_model(ModelType.TRANSFORMER, config=custom_config)
    print(transformer_model)
