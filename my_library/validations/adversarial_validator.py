import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class AdversarialValidator:
    """
    Detects distribution shift between training and test datasets using adversarial validation.

    This class trains a binary classifier to distinguish between train and test datasets.
    A high cross-validated score (e.g. AUC) indicates a significant distribution difference,
    which often suggests overfitting risk or real-world data drift.

    Note:
        This validator works for both classification and regression problems, since the task type
        only affects the downstream target—not the input features,
        which are what adversarial validation inspects.

    Attributes:
        cv (int): Number of folds for cross-validation.
        scoring (str): Scoring method for cross-validation (e.g., "roc_auc", "accuracy").
        random_state (int): Random seed for reproducibility.
        fitted_model (RandomForestClassifier): The trained model distinguishing train/test.
        feature_names (List[str]): Names of the features used.
        X (pd.DataFrame): Combined features from train and test, for internal validation use.
        y (np.ndarray): Binary labels indicating train (0) or test (1).
    """

    def __init__(self, cv: int = 5, scoring: str = "roc_auc", random_state: int = 42):
        """
        Initialize the adversarial validator.

        Args:
            cv (int): Number of folds for cross-validation.
            scoring (str): Scoring function for model evaluation.
            random_state (int): Seed to control shuffling and model reproducibility.
        """
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.fitted_model = None
        self.feature_names = None
        self.X = None
        self.y = None

    def _prepare_data(self, train_X: pl.DataFrame, test_X: pl.DataFrame) -> None:
        """
        Internal utility to concatenate and label train/test datasets.

        Args:
            train_X (pl.DataFrame): Training feature set.
            test_X (pl.DataFrame): Test feature set.

        Raises:
            ValueError: If the columns of train_X and test_X do not match.
        """
        if train_X.columns != test_X.columns:
            raise ValueError("Train and test must have the same columns in the same order.")

        train = train_X.with_columns(pl.lit(0).alias("is_test"))
        test = test_X.with_columns(pl.lit(1).alias("is_test"))

        full_data = pl.concat([train, test]).sample(fraction=1.0, seed=self.random_state)
        self.X = full_data.drop("is_test").to_pandas()
        self.y = full_data["is_test"].to_numpy()
        self.feature_names = list(self.X.columns)

    def fit(self, train_X: pl.DataFrame, test_X: pl.DataFrame) -> None:
        """
        Fit a binary classifier to distinguish between train and test data.

        Args:
            train_X (pl.DataFrame): Feature matrix for training data.
            test_X (pl.DataFrame): Feature matrix for test data.
        """
        self._prepare_data(train_X, test_X)
        self.fitted_model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        ).fit(self.X, self.y)

    def validate(self) -> float:
        """
        Evaluate how distinguishable the train and test datasets are using cross-validation.

        Returns:
            float: Cross-validated score (e.g., AUC). Higher values imply more distribution shift.

        Raises:
            RuntimeError: If `fit()` was not called before.
        """
        if self.X is None or self.y is None:
            raise RuntimeError("You must call `fit()` before `validate()`.")

        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            self.X, self.y, cv=self.cv, scoring=self.scoring
        )
        return float(np.mean(scores))

    def get_feature_importance(self, top_k: int = 20) -> pl.DataFrame:
        """
        Retrieve top-k features most important for distinguishing train and test sets.

        Args:
            top_k (int): Number of top features to return.

        Returns:
            pl.DataFrame: DataFrame with columns "feature" and "importance", sorted descending.

        Raises:
            RuntimeError: If model is not yet fitted.
        """
        if self.fitted_model is None:
            raise RuntimeError("You must call `fit()` before `get_feature_importance()`.")

        importances = self.fitted_model.feature_importances_
        return pl.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort("importance", descending=True).head(top_k)
