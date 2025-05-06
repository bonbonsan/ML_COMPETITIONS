import random
from typing import Any, Dict, Tuple, Union

import numpy as np

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.parameter_tunings.tuner_base import BaseTuner
from my_library.utils.timeit import timeit
from my_library.validations.validation_runner import ValidationRunner


class RandomSearchValidationTuner(BaseTuner):
    """
    Tuner that performs random search over the hyperparameter space
    using ValidationRunner for cross-validation.

    Samples random parameter sets from the defined search space, updates
    the model configuration, and evaluates performance over the provided folds.

    Inherits from BaseTuner.
    """
    @timeit
    def tune(
        self,
        fit_config: FitConfig,
        return_runner_results: bool = False
    ) -> Union[
        Tuple[Dict[str, Any], float],
        Tuple[Dict[str, Any], float, Dict[str, Any]]
    ]:
        """
        Execute random search hyperparameter optimization.

        Args:
            fit_config (FitConfig): Fit configuration (features, early stopping, epochs, etc.).
            return_runner_results (bool): If True, returns runner results dict
                in addition to best_params and best_score.

        Returns:
            Tuple of (best_params, best_score) or
            (best_params, best_score, best_runner_result).
        """
        keys = list(self.param_space.keys())
        all_choices = [
            self.param_space[k] if isinstance(self.param_space[k], list)
            else np.linspace(*self.param_space[k], num=10)
            for k in keys
        ]

        for _ in range(self.n_trials):
            trial_params = {k: random.choice(all_choices[i]) for i, k in enumerate(keys)}
            self.model_configs.params.update(trial_params)

            runner = ValidationRunner(
                model_class=self.model_class,
                model_configs=self.model_configs,
                metric_fn=self.scoring,
                predict_proba=self.predict_proba,
                parallel_mode=self.parallel_mode
            )
            result = runner.run(self.folds, fit_config)
            score = result["mean_score"]

            if self.best_score is None or \
               (self.maximize and score > self.best_score) or \
               (not self.maximize and score < self.best_score):
                self.best_score = score
                self.best_params = trial_params.copy()
                self.best_runner_result = result

        if return_runner_results:
            return self.best_params, self.best_score, self.best_runner_result
        else:
            return self.best_params, self.best_score
