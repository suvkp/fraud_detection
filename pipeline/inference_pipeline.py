import pandas as pd
import numpy as np
import logging
from src.data_preprocessing import DataPreprocessingTrain, DataPreprocessingInference
from src.data_loader import DataLoader
from src.model_training import ModelTraining
from src.model_inference import ModelInference
from src.model_evaluation import ModelEvaluation

def inference_pipeline(id_path: str, transaction_path: str, model_object: object) -> None:
    """
    Run the inference pipeline.
    Args:
        id_path (str): Path to the identity data.
        transaction_path (str): Path to the transaction data.
        model_object (object): Model object.
    Returns:
        None
    """

    # ingest the data
    test_id = DataLoader(id_path)
    test_trans = DataLoader(transaction_path)
    logging.info(f"identity data shape: {test_id.dataset.shape}")
    logging.info(f"transaction data shape: {test_trans.dataset.shape}")

    # preprocessing the test data
    X_test = DataPreprocessingInference().transform(test_id.dataset, test_trans.dataset)
    logging.info(f"X_test shape: {X_test.shape}")
    try:
        X_test.shape[0] == test_id.dataset.shape[0]
    except AssertionError:
        logging.error("The number of rows in X_test and test_id do not match.")

    # Make inference
    logging.info("Starting inference...")
    try:
        y_pred = ModelInference(model_object=model_object).predict_proba(X_test)
    except Exception as e:
        logging.error(f"Error during inference: {e}")

    # Save the predictions
    try:
        submission = pd.DataFrame({'TransactionID': test_id.dataset['TransactionID'], 'isFraud': y_pred})
        submission.to_csv('submission.csv', index=False)
        logging.info("Predictions saved to submission.csv")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")
    logging.info("Inference completed.")
    return None