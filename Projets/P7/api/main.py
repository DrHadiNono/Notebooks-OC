# -*- coding: utf-8 -*-

# Library imports
import uvicorn
from fastapi import FastAPI
from HomeCreditApplicants import HomeCreditApplicant
import dill
import codecs
import pandas as pd
import shap

# Create the app object
app = FastAPI()

# Read the model features
cols = []
with open("cols.txt", "r") as f:
    cols = f.read().split()

# Load the model
model = None
with open("model.pkl", "rb") as f:
    model = dill.load(f)

# Load the data
df = pd.read_csv('data-sample.csv', dtype={'SK_ID_CURR': 'int32',
                 'GENDER': 'int32', 'OWN_CAR': 'int32', 'OWN_REALTY': 'int32'})
df = df.drop(columns='TARGET')
X = df.drop(columns='SK_ID_CURR')
ids_list = df['SK_ID_CURR'].values.astype(int).tolist()

# Compute SHAP values
explainer = shap.TreeExplainer(model, X, model_output='probability')
expected_value = explainer.expected_value


def predict(applicant: HomeCreditApplicant):
    prediction = model.predict_proba([applicant.get_values()])[0]
    return prediction[1]


def dillEncode(data):
    return codecs.encode(dill.dumps(data), "base64").decode()


def dillDecode(data):
    return dill.loads(codecs.decode(data.encode(), "base64"))

# Index route, opens automatically on http://127.0.0.1:8000


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/data')
def data():
    return df.to_dict('tight')


@app.get('/ids')
def ids():
    return ids_list


# Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence

@app.post('/applicant')
def get_applicant(id: int):
    return HomeCreditApplicant(df[df['SK_ID_CURR'] == id].to_dict('records')[0])


@app.post('/scoreApplicant')
def scoreApplicant(applicant: str):
    applicant = HomeCreditApplicant(dillDecode(applicant))
    return predict(applicant)


@app.post('/scoreID')
def scoreID(id: int):
    applicant = get_applicant(id)
    return predict(applicant)


@app.get('/expectedValue')
def expectedValue():
    return dillEncode(expected_value)


@app.post('/shapExplanationID')
def shapExplanationID(id: int):
    applicant = df[df['SK_ID_CURR'] == id].drop(columns='SK_ID_CURR')
    return dillEncode(explainer(applicant)[0])


@app.post('/shapExplanationApplicant')
def shapExplanationApplicant(applicant: str):
    applicant = pd.DataFrame.from_dict(dillDecode(applicant))
    return dillEncode(explainer(applicant.drop(columns='SK_ID_CURR'))[0])


# Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload
