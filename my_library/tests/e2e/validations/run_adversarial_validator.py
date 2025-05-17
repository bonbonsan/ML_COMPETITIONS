import numpy as np
import polars as pl

from my_library.utils.data_loader import load_sample_data
from my_library.validations.adversarial_validator import AdversarialValidator

# 1. Load real iris data (assumed to be pandas)
iris_df = load_sample_data(name="iris", task="classification")

# 2. Drop target column if present
feature_cols = [col for col in iris_df.columns if col != "target"]
real_x_pd = iris_df[feature_cols]

# 3. Convert to Polars
real_x = pl.from_pandas(real_x_pd)

# 4. Generate dummy data with same column schema
np.random.seed(42)
dummy_data = {
    col: np.random.uniform(low=real_x[col].min(), high=real_x[col].max(), size=real_x.height)
    for col in real_x.columns
}
dummy_x = pl.DataFrame(dummy_data)

# 5. Run Adversarial Validation
validator = AdversarialValidator(scoring="roc_auc")
validator.fit(train_X=real_x, test_X=dummy_x)

score = validator.validate()
print(f"Adversarial AUC (real vs dummy): {score:.4f}")

# 6. Show most distinguishing features
print("Top features indicating distribution shift:")
print(validator.get_feature_importance())
