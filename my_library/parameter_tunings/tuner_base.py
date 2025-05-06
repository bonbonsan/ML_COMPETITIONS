from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.interface import CustomModelInterface


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuning of CustomModelInterface models.

    This class defines the interface and common behavior for tuning hyperparameters
    over cross-validation folds. Subclasses must implement the tune and save_best_model methods.

    Args:
        model_class (Callable[[Any], CustomModelInterface]): Factory for creating model instances.
        model_configs (ConfigBase): Base configuration object for the model.
        param_space (Dict[str, Any]): Dictionary defining the hyperparameter search space.
            Values can be lists of candidates or tuples (start, end) for continuous ranges.
        folds (List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]]): List of train/validation folds.
        scoring (Callable[[Any, Any], float]): Function to evaluate model predictions.
        early_stopping_rounds (Optional[int]): Rounds for early stopping during training.
        predict_proba (bool): Whether to use probability predictions for scoring.
        n_trials (int): Number of hyperparameter trials to perform.
        maximize (bool): If True, higher score is better; otherwise, lower score is better.
        parallel_mode (bool): If True, folds are processed in parallel (CPU only).
                            If False, folds are processed sequentially.

    Attributes:
        best_params (Optional[Dict[str, Any]]): Best hyperparameters found.
        best_score (Optional[float]): Best score achieved.
        best_runner_result (Optional[Dict[str, Any]]): Detailed results from validation runner.
    """
    def __init__(
        self,
        model_class: Callable[[Any], CustomModelInterface],
        model_configs: ConfigBase,
        param_space: Dict[str, Any],
        folds: List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]],
        scoring: Callable[[Any, Any], float],
        early_stopping_rounds: Optional[int] = None,
        predict_proba: bool = False,
        n_trials: int = 20,
        maximize: bool = True,
        parallel_mode: bool = False
    ):
        self.model_class = model_class
        self.model_configs = model_configs
        self.param_space = param_space
        self.folds = folds
        self.scoring = scoring
        self.early_stopping_rounds = early_stopping_rounds
        self.predict_proba = predict_proba
        self.n_trials = n_trials
        self.maximize = maximize
        self.parallel_mode = parallel_mode

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.best_runner_result: Optional[Dict[str, Any]] = None

    @abstractmethod
    def tune(
        self,
        fit_config: FitConfig,
        return_runner_results: bool = False
    ) -> Union[
        Tuple[Dict[str, Any], float],
        Tuple[Dict[str, Any], float, Dict[str, Any]]
    ]:
        """
        Run the hyperparameter tuning procedure.

        Must be implemented by subclasses.

        Args:
            fit_config (FitConfig): Configuration for fitting/training
                                    (features, early stopping, epochs, etc.).
            return_runner_results (bool): If True, also return detailed runner results.

        Returns:
            Tuple containing best_params and best_score,
            and optionally best_runner_result if return_runner_results is True.
        """
        pass

    def save_best_model(self, path: str):
        """
        Save the best model obtained during tuning to the specified path.

        Args:
            path (str): Filepath where the model will be saved.

        Raises:
            ValueError: If tuning has not been run or no result is available.
        """
        if self.best_runner_result is None:
            raise ValueError("No tuning result found. Run tune() before saving the model.")

        best_model = self.best_runner_result["fold_models"][0]
        best_model.save_model(path)
        print(f"Best model saved to: {path}")
