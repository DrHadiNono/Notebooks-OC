# -*- coding: utf-8 -*-

# Library imports
import uvicorn
from fastapi import FastAPI
from HomeCreditApplicants import HomeCreditApplicant
import dill
import pandas as pd

# Create the app object
app = FastAPI()

cols = []
with open("cols.txt", "r") as f:
    cols = f.read().split()

classifier = None
with open("model.pkl", "rb") as f:
    classifier = dill.load(f)

df = pd.read_csv('data-sample.csv')
ids = df['SK_ID_CURR'].values.astype(int).tolist()

# Index route, opens automatically on http://127.0.0.1:8000


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/ids')
def index():
    return ids

# Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence


@app.post('/applicant')
def get_applicant(id: int):
    return HomeCreditApplicant(df[df['SK_ID_CURR'] == id].to_dict('records')[0])


@app.post('/predict')
def predict_score(id: int):
    applicant = get_applicant(id)
    prediction = classifier.predict_proba([applicant.get_values()])[0]
    return prediction[1]


# Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload
