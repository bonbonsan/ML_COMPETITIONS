import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from my_library.configs.model_configs.base_configs import ConfigBase
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger
from my_library.validations.validator import Validator

logger = Logger(__name__, save_to_file=False).get_logger()


class ValidationRunner:
    """
    Flexible validation runner for holdout and cross-validation with extended utilities.

    Attributes:
        model_class: Class implementing CustomModelInterface.
        model_configs: Configuration object containing task_type, etc.
        metric_fn: Single evaluation function for run().
        predict_proba: Whether to use predict_proba for classification.
        return_labels: Convert probabilities to labels when True.
        binary_threshold: Threshold for binary classification.
        folds: Stored input folds after run().
        results: Dictionary returned by run().

    Example:
        ### Holdout (1 fold):
        folds = [
            ((X_train, y_train), (X_valid, y_valid))
        ]

        ### K-Fold / Time Series CV (multiple folds):
        folds = [
            ((X_train_fold1, y_train_fold1), (X_valid_fold1, y_valid_fold1)),
            ((X_train_fold2, y_train_fold2), (X_valid_fold2, y_valid_fold2)),
            ...
        ]

        runner = ValidationRunner(...)
        results = runner.run(folds)
    """

    def __init__(
        self,
        model_class: CustomModelInterface,
        model_configs: ConfigBase,
        metric_fn: Optional[Callable] = None,
        predict_proba: bool = False,
        return_labels: bool = True,
        binary_threshold: float = 0.5
    ):
        self.model_class = model_class
        self.model_configs = model_configs
        self.metric_fn = metric_fn
        self.predict_proba = predict_proba
        self.return_labels = return_labels
        self.binary_threshold = binary_threshold

        self.folds = None
        self.results = None

        logger.info(
            f"Initialized ValidationRunner | model={model_class.__name__} "
            f"task={model_configs.task_type}"
        )

    def run(
        self,
        folds: List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]],
        fit_config: FitConfig
    ) -> Dict[str, Union[List[float], float, List[dict], List[pd.Series], str]]:
        """
        Execute validation across provided folds.

        Args:
            folds: List of ((X_train, y_train), (X_valid, y_valid)) tuples.
            fit_config: Base FitConfig (feats, early_stopping_rounds, epochs, batch_size).

        Returns:
            Dictionary containing:
              - fold_scores: List of per-fold scores.
              - mean_score: Mean of fold_scores.
              - std_score: Std deviation of fold_scores.
              - fold_models: Trained model objects per fold.
              - fold_params: Model.get_params() per fold.
              - fold_predictions: Predictions (or probas) per fold as pd.Series.
              - metric_fn: Name of metric function used.
              - task_type: classification or regression.
        """
        logger.info("Validation run started.")
        self.folds = folds

        task_type = self.model_configs.task_type
        scores, preds, models, params = [], [], [], []

        for i, ((X_tr, y_tr), (X_val, y_val)) in enumerate(folds):
            logger.info(f"[Fold {i+1}] Train={X_tr.shape} Valid={X_val.shape}")
            model = self.model_class(self.model_configs)
            fold_fc = FitConfig(
                feats=fit_config.feats,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=fit_config.early_stopping_rounds,
                epochs=fit_config.epochs,
                batch_size=fit_config.batch_size
            )
            # use validator variable for readability
            validator = Validator(model)
            validator.train(X_tr, y_tr, fit_config=fold_fc)

            # Prediction logic
            if self.predict_proba and task_type == "classification":
                proba = validator.predict_proba(X_val)
                if self.return_labels:
                    if proba.shape[1] > 1:
                        y_pred = pd.Series(proba.values.argmax(axis=1), index=X_val.index)
                    else:
                        y_pred = pd.Series(
                            (proba > self.binary_threshold).astype(int).ravel(), index=X_val.index
                            )
                else:
                    y_pred = pd.DataFrame(proba, index=X_val.index)
            else:
                y_pred = validator.predict(X_val)

            # evaluate using the same validator instance
            score = validator.evaluate(y_val, y_pred, metric_fn=self.metric_fn)
            logger.info(f"[Fold {i+1}] Score={score:.4f}")

            scores.append(score)
            preds.append(y_pred)
            models.append(model)
            params.append(model.get_params())

        mean_score, std_score = float(np.mean(scores)), float(np.std(scores))
        logger.info(f"Validation completed | mean={mean_score:.4f} std={std_score:.4f}")

        self.results = {
            "fold_scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_models": models,
            "fold_params": params,
            "fold_predictions": preds,
            "metric_fn": self.metric_fn.__name__ if self.metric_fn else (
                "accuracy_score" if task_type == "classification" else "rmse"
            ),
            "task_type": task_type
        }
        return self.results

    def run_predict_on_test(
        self,
        trained_model: CustomModelInterface,
        X_test: pd.DataFrame,
        use_proba: Optional[bool] = None
    ) -> Union[np.ndarray, pd.Series]:
        """
        Predict on test data using a trained model.

        Args:
            trained_model: Already trained model.
            X_test: Test features.
            use_proba: Whether to return probabilities (only for classification).

        Returns:
            Predictions or predicted probabilities.
        """
        logger.info("run_predict_on_test invoked.")
        if use_proba is None:
            use_proba = self.predict_proba

        task_type = self.model_configs.task_type
        if use_proba and task_type == "classification":
            proba = trained_model.predict_proba(X_test)
            if self.return_labels:
                if proba.shape[1] > 1:
                    preds = proba.values.argmax(axis=1)
                else:
                    preds = (proba.values > self.binary_threshold).astype(int).ravel()
            else:
                preds = proba
        else:
            preds = trained_model.predict(X_test)

        logger.info(f"run_predict_on_test completed | outputs={len(preds)}")
        return preds
    
    def get_oof_predictions(self) -> pd.Series:
        """
        Gather out-of-fold predictions aligned with original indices.

        Returns:
            pd.Series: Combined OOF predictions indexed by original row index.
        """
        if not self.results or not self.folds:
            raise RuntimeError("run() must be called before get_oof_predictions().")
        # concatenate series from each fold
        oof = pd.concat(self.results["fold_predictions"], axis=0)
        return oof.sort_index()

    def get_meta_features(self) -> pd.DataFrame:
        """
        Build DataFrame of OOF predictions for stacking.

        Returns:
            pd.DataFrame: Single-column DataFrame named 'oof_pred'.
        """
        oof = self.get_oof_predictions()
        return pd.DataFrame({"oof_pred": oof})

    def fit_meta_model(
        self,
        meta_model: CustomModelInterface,
        fit_config: FitConfig
    ) -> None:
        """
        Train a meta_model on OOF predictions.

        Args:
            meta_model: Model implementing .fit(X, y, fit_config).
            fit_config: FitConfig for meta model training.
        """
        meta_X = self.get_meta_features()
        true_list = []
        for (_, (_, y_val)) in self.folds:
            true_list.append(y_val)
        meta_y = pd.concat(true_list, axis=0).sort_index()
        logger.info("Fitting meta model on OOF features.")
        meta_model.fit(meta_X, meta_y, fit_config=fit_config)

    def ensemble_predict(self, X_test: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Aggregate predictions from each fold model on new data.

        Args:
            X_test: Test feature DataFrame.

        Returns:
            pd.Series or pd.DataFrame: Averaged or majority-vote predictions.
        """
        if not self.results:
            raise RuntimeError("run() must be called before ensemble_predict().")
        task = self.results["task_type"]
        preds_list = []
        for model in self.results["fold_models"]:
            if task == "classification" and self.predict_proba:
                preds_list.append(model.predict_proba(X_test))
            else:
                preds_list.append(model.predict(X_test))
        arr = np.stack(preds_list, axis=0)
        if task == "regression":
            return pd.Series(arr.mean(axis=0), index=X_test.index)
        # classification
        if self.predict_proba:
            avg_proba = arr.mean(axis=0)
            if self.return_labels:
                if avg_proba.shape[1] > 1:
                    return pd.Series(avg_proba.argmax(axis=1), index=X_test.index)
                return pd.Series(
                    (avg_proba > self.binary_threshold).astype(int).ravel(), index=X_test.index
                    )
            return pd.DataFrame(avg_proba, index=X_test.index)
        # majority vote
        vote_preds = [np.bincount(arr[:, i].astype(int)).argmax() for i in range(arr.shape[1])]
        return pd.Series(vote_preds, index=X_test.index)

    def average_params(self) -> Dict[str, Union[float, str]]:
        """
        Compute average of numeric hyperparameters across folds.

        Returns:
            dict: Averaged numeric params, first value for non-numerics.
        """
        if not self.results:
            raise RuntimeError("run() must be called before average_params().")
        param_dicts = self.results["fold_params"]
        avg_params = {}
        for k in param_dicts[0].keys():
            vals = [d[k] for d in param_dicts]
            if all(isinstance(v, (int, float)) for v in vals):
                avg_params[k] = float(np.mean(vals))
            else:
                avg_params[k] = vals[0]
        return avg_params

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Summarize feature importances across folds via get_top_features.

        Returns:
            pd.DataFrame: Features index with columns ['mean','std'].
        """
        if not self.results or not self.folds:
            raise RuntimeError("run() must be called before get_feature_importance_summary().")
        # Establish consistent feature order using original DataFrame columns
        features = list(self.folds[0][0][0].columns)
        imp_mat = []
        for model in self.results["fold_models"]:
            # Align feature importance order: get_top_features returns sorted list,
            # we convert to dict and then extract values in 'features' order
            top_feats = dict(model.get_top_features(top_n=len(features)))  # Ordered by importance
            imp_mat.append([top_feats.get(f, 0.0) for f in features])
        arr = np.array(imp_mat).T
        return pd.DataFrame({'mean': arr.mean(axis=1), 'std': arr.std(axis=1)}, index=features)

    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Compute two-sided confidence interval for CV scores.

        Args:
            alpha: Significance level.

        Returns:
            (lower_bound, upper_bound)
        """
        if not self.results:
            raise RuntimeError("run() must be called before get_confidence_interval().")
        scores = np.array(self.results["fold_scores"])
        n = len(scores)
        mean, sd = scores.mean(), scores.std(ddof=1)
        margin = t.ppf(1 - alpha/2, df=n-1) * sd / np.sqrt(n)
        return mean - margin, mean + margin

    def plot_learning_curves(self) -> None:
        """
        Plot training vs. validation loss if model.evals_result_ exists.

        Raises:
            RuntimeError if run() not called.
        """
        if not self.results:
            raise RuntimeError("run() must be called before plot_learning_curves().")
        for i, model in enumerate(self.results["fold_models"]):
            if hasattr(model, 'evals_result_'):
                res = model.evals_result_
                epochs = range(len(res['train']))
                plt.figure()
                plt.plot(epochs, res['train'], label='train')
                plt.plot(epochs, res['validation'], label='validation')
                plt.title(f"Learning Curve Fold {i+1}")
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
            else:
                logger.warning(f"Fold {i+1}: No evals_result_ attribute.")
        plt.show()

    def run_multi_metrics(
        self,
        folds: List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]],
        fit_config: FitConfig,
        metric_fns: List[Callable]
    ) -> Dict[str, List[float]]:
        """
        Execute CV computing multiple metrics per fold.

        Args:
            folds: Same as run().
            fit_config: Same as run().
            metric_fns: List of evaluation functions.

        Returns:
            dict: {metric_name: [scores per fold]}
        """
        logger.info("Multi-metric run started.")
        scores = {fn.__name__: [] for fn in metric_fns}
        for ((X_tr, y_tr), (X_val, y_val)) in folds:
            model = self.model_class(self.model_configs)
            Validator(model).train(X_tr, y_tr, fit_config=fit_config)
            y_pred = model.predict(X_val)
            for fn in metric_fns:
                scores[fn.__name__].append(fn(y_val, y_pred))
        return scores

    # def calibrate_proba(self, method: str = "isotonic") -> List[CalibratedClassifierCV]:
    #     """
    #     Calibrate probability estimates of each fold model.

    #     Args:
    #         method: 'sigmoid' or 'isotonic'.

    #     Returns:
    #         List of fitted CalibratedClassifierCV instances.
    #     """
    #     if not self.results or self.model_configs.task_type != "classification":
    #         raise RuntimeError("Calibration only after classification run().")
    #     calibrated = []
    #     for i, (((X_tr, y_tr), (X_val, y_val)), model) in enumerate(
    #             zip(self.folds, self.results["fold_models"])):
    #         model.classes_ = np.unique(y_tr)
    #         model._estimator_type = "classifier"
    #         calib = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    #         try:
    #             # pass numpy arrays to fit()
    #             X_fit = X_val.values if hasattr(X_val, "values") else X_val
    #             y_fit = y_val.values.ravel() \
    #                 if hasattr(y_val, "values") else np.asarray(y_val).ravel()
    #             calib.fit(X_fit, y_fit)
    #         except ValueError as e:
    #             # skip failures but log warning
    #             logger.warning(f"[Calibration fold {i+1}] failed: {e}")
    #         calibrated.append(calib)
    #     return calibrated

    def export_cv_report(self, path: str, as_excel: bool = False) -> None:
        """
        Export CV results and scores to CSV or Excel.

        Args:
            path: Destination file path.
            as_excel: If True, save as .xlsx; else .csv.
        """
        if not self.results:
            raise RuntimeError("run() must be called before export_cv_report().")
        report = pd.DataFrame({
            'fold': list(range(1, len(self.results['fold_scores'])+1)),
            'score': self.results['fold_scores']
        })
        report['mean_score'] = self.results['mean_score']
        report['std_score'] = self.results['std_score']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if as_excel:
            report.to_excel(path, index=False)
        else:
            report.to_csv(path, index=False)
        logger.info(f"Exported CV report to {path}")
    
    # Future hooks: optuna_objective(), grid_search(), random_search(), hyperopt_tuner() など
    # 並行処理
    # ログ
