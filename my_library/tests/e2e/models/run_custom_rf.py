from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.rf_configs import RandomForestConfig
from my_library.models.custom_rf import CustomRandomForest
from my_library.utils.data_loader import load_sample_data

# ------------------------------
# Classification Task: Iris
# ------------------------------
print("\n--- Classification: Iris ---")
iris_df = load_sample_data(name="iris", task="classification")
X_iris = iris_df.drop(columns="target")
y_iris = iris_df["target"]

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# configure CustomRandomForest for classification
clf_config = RandomForestConfig(
    model_name="CustomRF_Iris",
    task_type="classification",
    save_log=False,
    params={
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    }
)

fit_config = FitConfig(
    feats=None,
    eval_set=None,
    early_stopping_rounds=None,
    epochs=None,
    batch_size=None
)

clf_model = CustomRandomForest(config=clf_config)
clf_model.fit(X_train, y_train, fit_config=fit_config)

# generate predictions and probabilities
preds = clf_model.predict(X_test)
proba = clf_model.predict_proba(X_test)

print("Predictions:", preds.values[:5])
print("Probabilities:\n", proba.head())

# evaluate accuracy
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

# show top-5 features by permutation importance
print("Top features:", clf_model.get_top_features(top_n=5))

print("Model summary:", clf_model)

# ------------------------------
# Regression Task: Diabetes
# ------------------------------
print("\n--- Regression: Diabetes ---")
diabetes_df = load_sample_data(name="diabetes", task="regression")
X_diab = diabetes_df.drop(columns="target")
y_diab = diabetes_df["target"]

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

# configure CustomRandomForest for regression
reg_config = RandomForestConfig(
    model_name="CustomRF_Diabetes",
    task_type="regression",
    save_log=False,
    params={
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    }
)

fit_config = FitConfig(
    feats=None,
    eval_set=None,
    early_stopping_rounds=None,
    epochs=None,
    batch_size=None
)

reg_model = CustomRandomForest(config=reg_config)
reg_model.fit(X_train, y_train, fit_config=fit_config)

# generate predictions
reg_preds = reg_model.predict(X_test)

# evaluate MSE
mse = mean_squared_error(y_test, reg_preds)
print(f"MSE: {mse:.4f}")

# show top-5 features by permutation importance
print("Top features:", reg_model.get_top_features(top_n=5))

print("Model summary:", reg_model)
