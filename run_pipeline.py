from pipeline.train_pipeline import train_pipeline
# from pipeline.inference_pipeline import inference_pipeline

if __name__ == "__main__":
    id_path = "data/train_identity.csv"
    transaction_path = "data/train_transaction.csv"
    train_pipeline(id_path, transaction_path)