import pandas as pd

def convert_to_datetime(df, date_columns=None, format=None, errors='raise'):
    """
    Converts specified columns in a DataFrame to datetime format.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_columns (list, optional): List of column names to convert. 
                                       If None, all object/string columns are considered.
        format (str, optional): The datetime format to use for conversion (e.g., '%Y-%m-%d').
                                If None, pandas will infer the format.
        errors (str, optional): How to handle conversion errors. 
                                'raise' will raise an exception,
                                'coerce' will replace invalid parsing with NaT,
                                'ignore' will leave the column unchanged.
    
    Returns:
        pd.DataFrame: DataFrame with converted datetime columns.
    """
    # If no specific columns are provided, infer object or string columns
    if date_columns is None:
        date_columns = df.select_dtypes(include=['object', 'string']).columns
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], format=format, errors=errors)
            print(f"Successfully converted column '{col}' to datetime.")
        except Exception as e:
            print(f"Failed to convert column '{col}' to datetime. Error: {e}")
    
    return df

def get_min_max(df, col_name):
    return print(f"Min: {min(df[col_name])}\nMax: {max(df[col_name])}")