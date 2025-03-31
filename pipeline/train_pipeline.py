import pandas as pd
import numpy as np
import logging
from src.data_preprocessing import DataPreprocessingTrain, DataPreprocessingInference
from src.data_loader import DataLoader
from src.model_training import ModelTraining
from src.model_inference import ModelInference
from src.model_evaluation import ModelEvaluation

def train_pipeline(id_path: str, transaction_path: str) -> None:
    """
    Run the training pipeline.
    Args:
        id_path (str): Path to the identity data.
        transaction_path (str): Path to the transaction data.
    Returns:
        None
    """
    # get the data
    train_id = DataLoader(id_path)
    train_trans = DataLoader(transaction_path)
    logging.info(f"identity data shape: {train_id.dataset.shape}")
    logging.info(f"transaction data shape: {train_trans.dataset.shape}")

    # preprocess the training data
    X_train, X_val, y_train, y_val = DataPreprocessingTrain(create_val_set=True).transform(train_id.dataset, train_trans.dataset)
    try:
        X_train.shape[0] == y_train.shape[0]
    except AssertionError:
        logging.error("The number of rows in X_train and y_train do not match.")
    try:
        X_val.shape[0] == y_val.shape[0]
    except AssertionError:
        logging.error("The number of rows in X_val and y_val do not match.")
    logging.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
    logging.info(f"y_train value counts: {y_train.value_counts()}")
    logging.info(f"y_val value counts: {y_val.value_counts()}")

    # get categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    logging.info(f"Categorical columns: {cat_cols}")

    # Model training
    params = {
        # 'iterations': 100,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42
    }
    fit_params = {
        'cat_features': cat_cols,
        'early_stopping_rounds': 25,
        'eval_set': (X_val, y_val),
        'verbose': 100,
        'use_best_model': True
    }
    logging.info("Starting model training...")
    trainer = ModelTraining(params=params)
    trainer.train(X_train, y_train, fit_params=fit_params)
    logging.info("Model training completed.")

    # Model evaluation
    logging.info("Starting model evaluation on validation set...")
    ModelEvaluation(model=trainer.model).evaluate(X_val, y_val, threshold=0.5)
    logging.info("Model evaluation completed.")
    return None

