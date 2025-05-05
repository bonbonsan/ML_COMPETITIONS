from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.tabn_configs import TabNetConfig
from my_library.models.custom_tabn import CustomTabNetModel
from my_library.utils.data_loader import load_sample_data

# ------------------------------
# Classification Task: Iris
# ------------------------------
print("\n--- Classification: Iris ---")
iris_df = load_sample_data(name="iris", task="classification")
X_iris = iris_df.drop(columns="target")
y_iris = iris_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

clf_config = TabNetConfig(
    model_name="TabNet_Iris",
    task_type="classification",
    use_gpu=False,
    save_log=False,
    params={
        "n_d": 16,
        "n_a": 16,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-3,
        "batch_size": 128,
        "virtual_batch_size": 32,
        "max_epochs": 100,
        "patience": 20,
        "lr": 2e-2,
        "verbose": 0,
        "seed": 42,
    }
)

fit_config = FitConfig(
        feats=None,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=None,
        epochs=100,
        batch_size=128
    )

clf_model = CustomTabNetModel(config=clf_config)
clf_model.fit(X_train, y_train, fit_config=fit_config)

preds = clf_model.predict(X_test)
proba = clf_model.predict_proba(X_test)
print("Predictions:", preds[:5].values)
print("Probabilities:\n", proba.head())

acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

print("Top features:")
print(clf_model.get_top_features(top_n=5))

# clf_model.save_model("my_library/test_model_iris.pkl")
# clf_model.load_model("my_library/test_model_iris.pkl")
print(clf_model)

# ------------------------------
# Regression Task: Diabetes
# ------------------------------
print("\n--- Regression: Diabetes ---")
diabetes_df = load_sample_data(name="diabetes", task="regression")
X_diab = diabetes_df.drop(columns="target")
y_diab = diabetes_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

reg_config = TabNetConfig(
    model_name="TabNet_Diabetes",
    task_type="regression",
    use_gpu=False,
    save_log=False,
    params={
        "n_d": 16,
        "n_a": 16,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-3,
        "batch_size": 128,
        "virtual_batch_size": 32,
        "max_epochs": 100,
        "patience": 20,
        "lr": 2e-2,
        "verbose": 0,
        "seed": 42,
    }
)

fit_config = FitConfig(
        feats=None,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=None,
        epochs=100,
        batch_size=128
    )

reg_model = CustomTabNetModel(config=reg_config)
reg_model.fit(X_train, y_train, fit_config=fit_config)

reg_preds = reg_model.predict(X_test)
mse = mean_squared_error(y_test, reg_preds)
print(f"MSE: {mse:.4f}")

print("Top features:")
print(reg_model.get_top_features(top_n=5))

# reg_model.save_model("my_library/test_model_diabetes.pkl")
# reg_model.load_model("my_library/test_model_diabetes.pkl")
print(reg_model)
