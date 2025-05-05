import itertools
from typing import Any, Dict, Tuple, Union

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.parameter_tunings.tuner_base import BaseTuner
from my_library.validations.validation_runner import ValidationRunner


class GridSearchValidationTuner(BaseTuner):
    """
    Tuner that performs grid search over the hyperparameter space
    using ValidationRunner for cross-validation.

    Iterates over all possible combinations in `param_space`, updates
    the model configuration, and evaluates performance over the provided folds.
    Inherits from BaseTuner.
    """

    def tune(
        self,
        fit_config: FitConfig,
        return_runner_results: bool = False
    ) -> Union[
        Tuple[Dict[str, Any], float],
        Tuple[Dict[str, Any], float, Dict[str, Any]]
    ]:
        """
        Execute grid search hyperparameter optimization.

        Args:
            fit_config (FitConfig):
                Fit configuration (features, early stopping rounds, epochs, etc.).
            return_runner_results (bool):
                If True, returns runner results dict in addition to best_params and best_score.

        Returns:
            (best_params, best_score) or
            (best_params, best_score, best_runner_result) if return_runner_results is True.
        """
        keys = list(self.param_space.keys())
        combinations = list(itertools.product(*self.param_space.values()))

        for combo in combinations:
            trial_params = dict(zip(keys, combo, strict=False))
            self.model_configs.params.update(trial_params)

            runner = ValidationRunner(
                model_class=self.model_class,
                model_configs=self.model_configs,
                metric_fn=self.scoring,
                predict_proba=self.predict_proba
            )
            # RandomSearch と同様の呼び出しシグネチャに合わせる
            result = runner.run(self.folds, fit_config)
            score = result["mean_score"]

            is_better = (
                self.best_score is None or
                (self.maximize and score > self.best_score) or
                (not self.maximize and score < self.best_score)
            )
            if is_better:
                self.best_score = score
                self.best_params = trial_params.copy()
                self.best_runner_result = result

        if return_runner_results:
            return self.best_params, self.best_score, self.best_runner_result
        return self.best_params, self.best_score
