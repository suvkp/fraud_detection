import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

class ModelTraining:
    def __init__(self, params=None):
        self.model = CatBoostClassifier(**params)

    def train(self, X, y, fit_params=None):
        """Trains a model."""
        self.model.fit(X, y, **fit_params)
        return self