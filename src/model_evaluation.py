import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from .model_inference import ModelInference

class ModelEvaluation(ModelInference):
    def __init__(self, model):
        """Initializes the ModelEvaluation class."""
        self.model = model

    def evaluate(self, X, y, threshold=0.5):
        """Evaluates the model."""
        pred_proba = super().predict_proba(X)
        pred_labels = super().predict(X, threshold=threshold)
        print('classification_report:\n', classification_report(y, pred_labels))