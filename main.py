# filename: main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI(title="Jacket Size Prediction API")

# Load data
df = pd.read_csv("jacket_sizes_data.csv")

# Features and label
X = df[['Age', 'Height_cm', 'Weight_kg']]
y = df['Jacket_Size_in']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SGDRegressor(max_iter=2000, learning_rate='invscaling', eta0=0.005)
model.fit(X_scaled, y)

# Save model and scaler (for reuse)
joblib.dump(model, "jacket_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Request body for API
class JacketInput(BaseModel):
    age: float
    height_cm: float
    weight_kg: float

@app.post("/predict")
def predict_jacket_size(data: JacketInput):
    # Load saved model and scaler
    model = joblib.load("jacket_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Prepare input
    X_input = [[data.age, data.height_cm, data.weight_kg]]
    X_scaled_input = scaler.transform(X_input)

    # Predict
    predicted_size = model.predict(X_scaled_input)[0]

    return {
        "Predicted_Jacket_Size_in": round(predicted_size, 2),
        "Input": data.dict()
    }

