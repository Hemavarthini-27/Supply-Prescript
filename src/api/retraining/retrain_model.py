import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# Load original training data
dataset = pd.read_csv("data/processed/supply_chain_dataset.csv")

# Load preprocessor
preprocessor = joblib.load("models/xgboost_preprocessor.pkl")

# Target column
X = dataset.drop(columns=["Late_delivery_risk"])
y = dataset["Late_delivery_risk"]

# Transform features
X_processed = preprocessor.transform(X)

# Train model
model = XGBClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_processed, y)

# Save updated model
joblib.dump(model, "models/xgboost_delay_model.pkl")

print("✅ Model retrained successfully!")