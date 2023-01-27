from fastapi import FastAPI, File
from starlette.responses import Response
import json
from fastapi.middleware.cors import CORSMiddleware
from json import dumps, loads
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="API for Machine learning model",
    description="yoo",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Inputs(BaseModel):
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int

@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')

@app.post("/ml-model")
async def ml_model(values: Inputs):
    data=values.dict()
    # data = json.load(values)
    print(data)
    
    model = joblib.load("static\pricepredict.pkl")
    y=model.predict([[data['a'],data['b'],data['c'],data['d'],data['e'],data['f']]]).tolist()
    print(y)
    return {
        'Result' : y[0]
    }
