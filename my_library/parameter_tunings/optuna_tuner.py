from typing import Any, Dict, Tuple, Union

import optuna

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.parameter_tunings.tuner_base import BaseTuner
from my_library.utils.timeit import timeit
from my_library.validations.validation_runner import ValidationRunner


class OptunaValidationTuner(BaseTuner):
    """
    Tuner that performs hyperparameter optimization using Optuna.
    Samples parameter sets from the defined search space, updates
    the model configuration, and evaluates performance over the
    provided folds using ValidationRunner.

    Inherits from BaseTuner.
    """

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna study.

        Args:
            trial (optuna.Trial): Optuna trial object used to suggest hyperparameters.

        Returns:
            float: Trial score (negated if minimizing).
        """
        # Suggest hyperparameters
        trial_params = {
            key: trial.suggest_float(key, *self.param_space[key])
            if isinstance(self.param_space[key], tuple)
            else trial.suggest_categorical(key, self.param_space[key])
            for key in self.param_space
        }

        # Update model configs
        self.model_configs.params.update(trial_params)

        # Run validation
        runner = ValidationRunner(
            model_class=self.model_class,
            model_configs=self.model_configs,
            metric_fn=self.scoring,
            predict_proba=self.predict_proba,
            parallel_mode=self.parallel_mode
        )
        result = runner.run(self.folds, self._fit_config)
        score = result["mean_score"]

        # Track best raw score
        if self.best_score is None or \
           (self.maximize and score > self.best_score) or \
           (not self.maximize and score < self.best_score):
            self.best_score = score
            self.best_params = trial_params.copy()
            self.best_runner_result = result

        # Return objective value (negated if minimizing)
        return score if self.maximize else -score

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
        Execute Optuna-based hyperparameter optimization.

        Args:
            fit_config (FitConfig): Configuration for fitting/training
                                    (features, early stopping, epochs, etc.).
            return_runner_results (bool): If True, also returns the full runner results.

        Returns:
            Tuple of (best_params, best_score) or
            (best_params, best_score, best_runner_result).
        """
        # Store fit configuration for validation
        self._fit_config = fit_config

        # Create study and optimize
        study = optuna.create_study(
            direction="maximize" if self.maximize else "minimize"
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        # best_score already tracked as raw mean_score; update best_params
        self.best_params = study.best_params

        if return_runner_results:
            return self.best_params, self.best_score, self.best_runner_result
        else:
            return self.best_params, self.best_score
