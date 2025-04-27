from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import asyncio
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Load the model and top features separately
model_save_path = os.path.join("models", "kidney_disease_model.pkl")
features_save_path = os.path.join("models", "top_features.pkl")

# Load model and top features
pipeline = joblib.load(model_save_path)  # Load the trained model pipeline
top_features = joblib.load(features_save_path)  # Load the top features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KidneyDiseaseInput(BaseModel):
    SerumCreatinine: float
    GFR: float
    ProteinInUrine: float
    FastingBloodSugar: float
    BUNLevels: float
    HbA1c: float
    SerumElectrolytesSodium: float
    SystolicBP: float
    HemoglobinLevels: float

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    await asyncio.sleep(0.1)
    return {"message": "Server is running"}

@app.post("/predict")
def predict(data: KidneyDiseaseInput):
    input_data = [
        data.SerumCreatinine,
        data.GFR,
        data.ProteinInUrine,
        data.FastingBloodSugar,
        data.BUNLevels,
        data.HbA1c,
        data.SerumElectrolytesSodium,
        data.SystolicBP,
        data.HemoglobinLevels
    ]

    # Create input dataframe with the top features
    input_df = pd.DataFrame([input_data], columns=top_features)

    # Make a prediction
    prediction = pipeline.predict(input_df)[0]
    confidence = pipeline.predict_proba(input_df)[0][prediction] * 100

    # Return the prediction and confidence
    if prediction == 1:
        return {
            "prediction": "Chronic Kidney Disease Detected ⚠️",
            "confidence": f"{confidence:.2f}%"
        }
    else:
        return {
            "prediction": "No Chronic Kidney Disease ✅",
            "confidence": f"{confidence:.2f}%"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




