from typing import Any, Dict, Tuple, Union

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.parameter_tunings.tuner_base import BaseTuner
from my_library.utils.timeit import timeit
from my_library.validations.validation_runner import ValidationRunner


class HyperoptValidationTuner(BaseTuner):
    """
    Tuner that performs hyperparameter optimization using Hyperopt's Tree-structured
    Parzen Estimator (TPE) algorithm with cross-validation via ValidationRunner.

    Samples parameter sets according to the defined search space, converts integer parameters,
    updates the model configuration, and evaluates performance over the provided folds.

    Inherits from BaseTuner.
    """

    def _objective(
        self,
        params: Dict[str, Any],
        fit_config: FitConfig
    ) -> Dict[str, Any]:
        """
        Objective function for Hyperopt trials.

        Casts integer parameters, runs cross-validation, and tracks the best score and params.

        Args:
            params (Dict[str, Any]): Candidate hyperparameters for this trial.
            fit_config (FitConfig): Fit configuration (features, early stopping, epochs, etc.).

        Returns:
            Dict[str, Any]: Dictionary containing 'loss' and 'status' for Hyperopt.
        """
        # Cast integer-valued parameters to int
        casted = {}
        for key, val in params.items():
            if isinstance(self.param_space.get(key), tuple):
                casted[key] = int(val)
            else:
                casted[key] = val

        # Update model parameters
        self.model_configs.params.update(casted)

        # Initialize and run cross-validation
        runner = ValidationRunner(
            model_class=self.model_class,
            model_configs=self.model_configs,
            metric_fn=self.scoring,
            predict_proba=self.predict_proba,
            parallel_mode=self.parallel_mode
        )
        result = runner.run(self.folds, fit_config)
        score = result['mean_score']

        # Track best parameters and score
        if (
            self.best_score is None or
            (self.maximize and score > self.best_score) or
            (not self.maximize and score < self.best_score)
        ):
            self.best_score = score
            self.best_params = casted.copy()
            self.best_runner_result = result

        # Hyperopt minimizes the objective, so negate if maximizing
        loss = -score if self.maximize else score
        return {'loss': loss, 'status': STATUS_OK}

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
        Execute hyperparameter optimization over the defined search space
        using Hyperopt and ValidationRunner.

        Args:
            fit_config (FitConfig): Fit configuration (features, early stopping, epochs, etc.).
            return_runner_results (bool): If True, returns runner results dict
                in addition to best_params and best_score.

        Returns:
            Tuple of (best_params, best_score) or
            (best_params, best_score, best_runner_result).
        """
        # Construct Hyperopt search space with integer sampling for tuples
        space: Dict[str, Any] = {}
        for key, bounds in self.param_space.items():
            if isinstance(bounds, tuple):  # integer range
                space[key] = hp.quniform(key, bounds[0], bounds[1], 1)
            else:  # discrete choices
                space[key] = hp.choice(key, bounds)

        trials = Trials()
        # Optimize using TPE algorithm
        fmin(
            fn=lambda params: self._objective(params, fit_config),
            space=space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials
        )

        if return_runner_results:
            return self.best_params, self.best_score, self.best_runner_result
        return self.best_params, self.best_score
