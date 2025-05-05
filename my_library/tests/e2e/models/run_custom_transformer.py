from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.transformer_configs import TransformerConfig
from my_library.models.custom_transformer import CustomTransformerModel
from my_library.utils.data_loader import load_sample_data
from my_library.utils.preprocessing import create_sequences

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

clf_config = TransformerConfig(
    model_name="Transformer_Iris",
    task_type="classification",
    use_gpu=False,
    save_log=False,
    params={
        "input_size": X_train.shape[1],
        "output_size": y_train.nunique(),  # 3 classes
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "activation": "relu",
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 20,
        "time_steps": 1
    }
)

fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

clf_model = CustomTransformerModel(config=clf_config)
clf_model.fit(X_train, y_train, fit_config=fit_config)

preds = clf_model.predict(X_test)
proba = clf_model.predict_proba(X_test)

print("Predictions:", preds[:5].values)
print("Probabilities:\n", proba.head())

acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

print("Top features:")
print(clf_model.get_top_features(top_n=5))

# clf_model.save_model("my_library/test_transformer_model_iris.pth")
# clf_model.load_model("my_library/test_transformer_model_iris.pth")
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

reg_config = TransformerConfig(
    model_name="Transformer_Diabetes",
    task_type="regression",
    use_gpu=False,
    save_log=False,
    params={
        "input_size": X_train.shape[1],
        "output_size": 1,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "activation": "relu",
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 30,
        "time_steps": 1
    }
)
fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

reg_model = CustomTransformerModel(config=reg_config)
reg_model.fit(X_train, y_train, fit_config=fit_config)

reg_preds = reg_model.predict(X_test)
mse = mean_squared_error(y_test, reg_preds)
print(f"MSE: {mse:.4f}")

print("Top features:")
print(clf_model.get_top_features(top_n=5))

# reg_model.save_model("my_library/test_transformer_model_diabetes.pth")
# reg_model.load_model("my_library/test_transformer_model_diabetes.pth")
print(reg_model)

# ------------------------------
# Regression Task: Airline
# ------------------------------
print("\n--- Regression: Airline Passengers ---")
df_airline = load_sample_data(name="airline", task="regression")
df_airline.set_index("Month", inplace=True)

# シーケンス化
X_seq, y_seq = create_sequences(df_airline, col="Passengers", time_steps=5)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

reg_config_airline = TransformerConfig(
    model_name="Transformer_Airline",
    task_type="regression",
    use_gpu=False,
    save_log=False,
    params={
        "input_size": X_train.shape[1],
        "output_size": 1,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "activation": "relu",
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 50,
        "time_steps": 1  # ここは1でOK。既にシーケンス化されているので。
    }
)

reg_model_airline = CustomTransformerModel(config=reg_config_airline)
reg_model_airline.fit(X_train, y_train, fit_config=fit_config)

reg_preds_airline = reg_model_airline.predict(X_test)
mse_airline = mean_squared_error(y_test, reg_preds_airline)
print(f"MSE: {mse_airline:.4f}")

print("Top features:")
print(reg_model.get_top_features(top_n=5))

# reg_model_airline.save_model("my_library/test_transformer_model_airline.pth")
# reg_model_airline.load_model("my_library/test_transformer_model_airline.pth")
print(reg_model_airline)
