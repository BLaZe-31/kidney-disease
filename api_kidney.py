from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import asyncio
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load(os.path.join("models", "kidney_disease_model.pkl"))



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
    input_data = np.array([[ 
    data.SerumCreatinine,
    data.GFR,
    data.ProteinInUrine,
    data.FastingBloodSugar,
    data.BUNLevels,
    data.HbA1c,
    data.SerumElectrolytesSodium,
    data.SystolicBP,
    data.HemoglobinLevels
]])


    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction] * 100

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
