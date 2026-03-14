
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Insurance Fraud Detection API")

with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

class ClaimInput(BaseModel):
    features: dict

@app.get("/")
def root():
    return {"message": "Insurance Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: ClaimInput):
    try:
        input_df = pd.DataFrame([data.features])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
        return {
            "prediction": int(prediction),
            "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
