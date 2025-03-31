import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from .model_inference import ModelInference

class ModelEvaluation:
    def __init__(self, model):
        """Initializes the ModelEvaluation class."""
        self.model = model

    def evaluate(self, X, y, threshold=0.5):
        """Evaluates the model."""
        predictor = ModelInference(self.model)
        pred_proba = predictor.predict_proba(X)
        pred_labels = predictor.predict(X, threshold=threshold)
        temp = pd.concat([y, pd.Series(pred_proba[:,1])], axis=1)
        temp.columns = ['y', 'pred_proba']
        print('classification_report:\n', classification_report(y, pred_labels))
        print('\n Prediction probability (> threshold) distribution when y=1:')
        print(temp[temp['y'] == 1]['pred_proba'].describe())
        print('\n Prediction probability (> threshold) distribution when y=0:')
        print(temp[temp['y'] == 0]['pred_proba'].describe())