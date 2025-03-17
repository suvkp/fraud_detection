import pandas as pd
import numpy as np

class ModelInference:
    def __init__(self, model_object: object):
        self.model_object = model_object

    def predict_proba(self, X):
        """Make predictions using a trained model."""
        prediction = self.model_object.predict_proba(X)
        return prediction
    
    def predict(self, X, threshold=0.5):
        """Make predictions using a trained model."""
        prediction = self.predict_proba(X)
        prediction_label = np.where(prediction[:,1] >= threshold, 1, 0)
        return prediction_label