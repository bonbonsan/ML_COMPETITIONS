from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.rnn_configs import RNNConfig
from my_library.models.custom_rnn import CustomRNNModel
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

clf_config = RNNConfig(
    model_name="RNN_Iris",
    task_type="classification",
    use_gpu=False,
    save_log=False,
    params={
        "input_size": X_train.shape[1],  # T=1 の前提
        "time_steps": 1,
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": y_train.nunique(),
        "activation": None,
        "dropout": 0.0,
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 20
    }
)

fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

clf_model = CustomRNNModel(config=clf_config)
clf_model.fit(X_train, y_train, fit_config=fit_config)

preds = clf_model.predict(X_test)
proba = clf_model.predict_proba(X_test)

print("Predictions:", preds[:5].values)
print("Probabilities:\n", proba.head())

acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

print("Top features:")
print(clf_model.get_top_features(top_n=5))

# clf_model.save_model("my_library/test_rnn_model_iris.pth")
# clf_model.load_model("my_library/test_rnn_model_iris.pth")
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

reg_config = RNNConfig(
    model_name="RNN_Diabetes",
    task_type="regression",
    use_gpu=False,
    save_log=False,
    params={
        "input_size": X_train.shape[1],  # T=1 の前提
        "time_steps": 1,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
        "activation": None,
        "dropout": 0.1,
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 30
    }
)

fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

reg_model = CustomRNNModel(config=reg_config)
reg_model.fit(X_train, y_train, fit_config=fit_config)

reg_preds = reg_model.predict(X_test)
mse = mean_squared_error(y_test, reg_preds)
print(f"MSE: {mse:.4f}")

print("Top features:")
print(reg_model.get_top_features(top_n=5))

# reg_model.save_model("my_library/test_rnn_model_diabetes.pth")
# reg_model.load_model("my_library/test_rnn_model_diabetes.pth")
print(reg_model)
