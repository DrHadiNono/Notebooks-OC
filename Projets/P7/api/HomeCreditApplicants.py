# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from pydantic import BaseModel
# Class which describes Home Credit Applicants


class HomeCreditApplicant(BaseModel):
    features = OrderedDict()

    # default constructor
    def __init__(self, data={}):
        super().__init__()
        # Retreive all the columns used as features in the model
        cols = []
        with open("cols.txt", "r") as f:
            cols = f.read().split()

        if data == {}:
            # Fill the features with random values first
            for col in cols:
                self.features[col] = np.random.uniform(0, 1)
        else:
            for col in cols:
                self.features[col] = data[col]

    def get_values(self):
        values = []
        for key, value in self.features.items():
            if key not in ['SK_ID_CURR', 'TARGET']:
                values.append(value)
        return values
