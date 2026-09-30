import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Features (Input)
X = data.drop("math score", axis=1)

# Target (Output)
y = data["math score"]

# Convert text columns into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include="object").columns)
    ],
    remainder="passthrough"
)

# Machine Learning Model
model = RandomForestRegressor(random_state=42)

# Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train Model
pipeline.fit(X, y)

# Save Model
joblib.dump(pipeline, "model.pkl")

print("✅ Model trained successfully!")
print("✅ model.pkl created.")