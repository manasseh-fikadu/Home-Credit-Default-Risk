import joblib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('./catboost_reduced.joblib')
test_set = joblib.load('./x_test.joblib')

@app.get('/')
def get_home():
    return {'message': 'API top level working'}


@app.get('/random/')
def predict():
    random_row = test_set.sample(n=1)

    prediction = model.predict(random_row)

    probability = model.predict_proba(random_row)

    if np.isscalar(prediction):
        prediction = prediction.tolist()
    else:
        prediction = prediction[0].tolist()

    return {'prediction': prediction, 'probability': round(probability[0][1], 2) * 100, 'data': random_row.to_dict('records')}

