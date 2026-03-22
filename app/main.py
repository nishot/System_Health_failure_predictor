from fastapi import FastAPI
import pickle
from ML.Train.predict import predict_failure
from . import schema
app=FastAPI()
from fastapi.middleware.cors import CORSMiddleware

model=pickle.load(open("ML/Train/model.pkl",'rb'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return "Welcome TO System Failure predictor"



@app.post("/predict")
def predict_api(data:schema.systemData):
    input_values = list(data.model_dump().values())
    result=predict_failure(model,input_values)

    return result
