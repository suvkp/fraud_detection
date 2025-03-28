import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessingTrain:
    def __init__(self, create_val_set=False):
        """Initializes the DataPreprocessing class."""
        self.create_val_set = create_val_set
    
    def transform(self, id_df: pd.DataFrame, trans_df: pd.DataFrame):
        """Transforms the DataFrame."""
        # Merge the DataFrames
        merged_df = pd.merge(id_df, trans_df, on='TransactionID', how='right')
        # Remove features with > 90% missing values, id & date columns
        features_to_rem_id = ['id_07', 'id_08'] + [f'id_{i}' for i in range(21,28)] 
        features_to_rem_trans = ['dist1', 'D11'] + [f'M{i}' for i in range(1,10)] + [f'V{i}' for i in range(1,12)] + ['TransactionID', 'TransactionDT']
        col_to_rem = features_to_rem_id + features_to_rem_trans
        merged_df.drop(col_to_rem, axis=1, inplace=True)
        # Impute missing values
        preprocessed_df = self._impute_missing_values(merged_df)
        # Split the DataFrame into features and target
        X, y = self._Xy_split(preprocessed_df, target_col='isFraud')
        # Create validation set
        if self.create_val_set:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            return X_train, X_val, y_train, y_val
        return X, y

    def _check(self, df: pd.DataFrame, columns: list):
        """Checks if the columns are present in the DataFrame."""
        for column in columns:
            if column not in df.columns:
                raise ValueError(f'{column} not found in DataFrame.')

    def _impute_missing_values(self, df):
        """Imputes missing values in a given column."""
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                df[column].fillna(-999, inplace=True)
            else:
                df[column].fillna('<UKNWN>', inplace=True)
        return df

    def _Xy_split(self, df, target_col='isFraud'):
        """Splits the DataFrame into features and target."""
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        return X, y
    
class DataPreprocessingInference:
    def __init__(self):
        pass
    
    def transform(self, id_df: pd.DataFrame, trans_df: pd.DataFrame):
        """Transforms the DataFrame."""
        # rename columns
        id_df.columns = self._rename_columns(id_df.columns)
        # Merge the DataFrames
        merged_df = pd.merge(id_df, trans_df, on='TransactionID', how='right')
        # Remove features with > 90% missing values
        features_to_rem_id = ['id_07', 'id_08'] + [f'id_{i}' for i in range(21,28)]
        features_to_rem_trans = ['dist1', 'D11'] + [f'M{i}' for i in range(1,10)] + [f'V{i}' for i in range(1,12)] + ['TransactionID', 'TransactionDT']
        col_to_rem = features_to_rem_id + features_to_rem_trans
        merged_df.drop(col_to_rem, axis=1, inplace=True)
        # Impute missing values
        preprocessed_df = self._impute_missing_values(merged_df)
        return preprocessed_df

    def _check(self, df: pd.DataFrame, columns: list):
        """Checks if the columns are present in the DataFrame."""
        for column in columns:
            if column not in df.columns:
                raise ValueError(f'{column} not found in DataFrame.')

    def _impute_missing_values(self, df):
        """Imputes missing values in a given column."""
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                df[column].fillna(-999, inplace=True)
            else:
                df[column].fillna('<UKNWN>', inplace=True)
        return df
    
    def _rename_columns(self, colnames):
        """Finds hyphen in the column names and replaces with underscore."""
        new_colnames = []
        for colname in colnames:
            if '-' in colname:
                new_colname = colname.replace('-', '_')
                new_colnames.append(new_colname)
            else:
                new_colnames.append(colname)
        return new_colnames
        