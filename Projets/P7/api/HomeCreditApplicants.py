# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from pydantic import BaseModel
# Class which describes Home Credit Applicants


class HomeCreditApplicant(BaseModel):
    features = OrderedDict()

    # default constructor
    def __init__(self):
        super().__init__()
        # Retreive all the columns used as features in the model
        cols = []
        with open("cols.txt", "r") as f:
            cols = f.read().split()

        # Fill the features with random values first
        for col in cols:
            self.features[col] = np.random.uniform(0, 1)

    def get_values(self):
        values = []
        for _, value in self.features.items():
            values.append(value)
        return values


# test = HomeCreditApplicant()
# print(test.features)
# print(test.get_values())
