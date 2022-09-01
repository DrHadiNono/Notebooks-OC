# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from HomeCreditApplicants import HomeCreditApplicant
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("lgbm.pkl", "rb")
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000


@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence

@app.post('/predict')
def predict_score(data: HomeCreditApplicant):
    # data = data.dict()
    # variance = data['variance']
    # skewness = data['skewness']
    # curtosis = data['curtosis']
    # entropy = data['entropy']

    data = HomeCreditApplicant()
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict_prob([data.get_values()])
    # if(prediction[0] > 0.5):
    #     prediction = "Fake note"
    # else:
    #     prediction = "Its a Bank note"
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload
