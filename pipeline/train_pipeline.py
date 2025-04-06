import pandas as pd
import numpy as np
import logging
import mlflow
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

    # Set up logging
    log_file_path = 'log/taining_pipeline.log'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting training pipeline...")

    # get the data
    train_id = DataLoader(id_path)
    train_trans = DataLoader(transaction_path)
    logging.info(f"identity data shape: {train_id.dataset.shape}")
    logging.info(f"transaction data shape: {train_trans.dataset.shape}")

    # preprocess the training data
    X_train, X_val, y_train, y_val = DataPreprocessingTrain(create_val_set=True).transform(train_id.dataset, train_trans.dataset)
    
    # check for data consistency
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

    # check for missing values
    if X_train.isnull().sum().sum() > 0:
        logging.error("There are missing values in the training data.")
    if X_val.isnull().sum().sum() > 0:
        logging.error("There are missing values in the validation data.")

    # get categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    logging.info(f"Categorical columns: {cat_cols}")

    # Model training
    params = {
        # 'iterations': 100,
        'loss_function': 'Logloss',
        'custom_metric': ['Logloss', 'AUC'],
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

    # experiment tracking with MLflow
    mlflow.set_experiment("Fraud Detection")
    with mlflow.start_run():
        # Log model
        mlflow.log_param("model_type", "CatBoost")
        mlflow.log_param("params", params)
        mlflow.log_param("fit_params", fit_params)
        mlflow.catboost.log_model(trainer.model, artifact_path="model", input_example=X_val.iloc[:5])
        logging.info("Model logged to MLflow.")
        # Log metrics
        mlflow.log_metric("Logloss", trainer.model.get_best_score()['validation']['Logloss'])
        mlflow.log_metric("AUC", trainer.model.get_best_score()['validation']['AUC'])
        # # Log feature importance
        # feature_importance = trainer.model.get_feature_importance()
        # feature_names = X_train.columns.tolist()
        # feature_importance_df = pd.DataFrame({
        #     'feature': feature_names,
        #     'importance': feature_importance
        # })
        # feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
        # mlflow.log_artifact(feature_importance_df.to_csv(index=False), artifact_path="feature_importance.csv")
        # logging.info("Feature importance logged to MLflow.")

    return None

