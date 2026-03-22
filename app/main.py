from fastapi import FastAPI
import pickle
from ML.Train.predict import predict_failure
from . import schema
app=FastAPI()


model=pickle.load(open("ML/Train/model.pkl",'rb'))


@app.get("/")
def root():
    return "Welcome TO System Failure predictor"



@app.post("/")
def predict_api(data:schema.systemData):
    input_values = list(data.model_dump().values())
    result=predict_failure(model,input_values)

    return result
