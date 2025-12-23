from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_no_show

app = FastAPI(title="Patient No-Show Prediction API")

class AppointmentRequest(BaseModel):
    Gender: str
    Age: int
    Neighbourhood: str
    OnGovtWelfareBenefits: int
    Hypertension: int
    Diabetes: int
    Alcoholism: int
    Handicapped: int
    SMS_received: int
    ScheduledDay: str
    AppointmentDay: str

@app.post("/predict")
def predict(request: AppointmentRequest):
    return predict_no_show(request.dict())
