import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def missing_values(self, column: str):
        """Returns the number and percentage of missing values in a given column."""
        missing_count = self.df[column].isna().sum()
        total_count = len(self.df)
        missing_percentage = (missing_count / total_count) * 100
        return missing_count, missing_percentage
    
    def scatter_matrix_plot(self, columns=None):
        """Plots scatter matrix for selected numerical variables."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        sns.pairplot(self.df[columns])
        plt.show()
    
    def frequency_plot(self, columns=None, label_column=None):
        """Plots frequency plot for selected categorical variables and displays a table with the number of categories per variable."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
        category_counts = {}
        
        for col in columns:
            plt.figure(figsize=(10, 5))
            value_counts = self.df[col].value_counts()
            category_counts[col] = len(value_counts)
            
            if len(value_counts) > 10:
                value_counts = value_counts[:10]
            
            if label_column and label_column in self.df.columns:
                sns.countplot(data=self.df, x=col, hue=label_column, order=value_counts.index)
            else:
                sns.barplot(x=value_counts.index, y=value_counts.values)
            
            plt.xticks(rotation=45)
            plt.title(f'Frequency plot for {col}')
            plt.show()
        
        category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Variable', 'Number of Categories'])
        print(category_counts_df)
    
    def correlation_matrix(self):
        """Returns and plots the correlation matrix for numerical variables."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        return corr_matrix
    
    def target_distribution(self, target_column: str, task_type: str):
        """Plots target variable distribution based on the type of task."""
        plt.figure(figsize=(10, 5))
        if task_type.lower() == 'regression':
            sns.histplot(self.df[target_column], kde=True)
            plt.title(f'Distribution of {target_column}')
        elif task_type.lower() == 'classification':
            value_counts = self.df[target_column].value_counts(normalize=True) * 100
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Frequency plot for {target_column}')
            for i, v in enumerate(value_counts.values):
                plt.text(i, v + 1, f'{v:.2f}%', ha='center')
        plt.show()
    
    def dataframe_shape_and_unique_identifier(self):
        """Prints the shape of the dataframe and detects the column uniquely identifying records."""
        shape = self.df.shape
        unique_id_column = None
        
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df):
                unique_id_column = col
                break
        
        print(f'DataFrame Shape: {shape}')
        if unique_id_column:
            print(f'Unique Identifier Column: {unique_id_column}')
        else:
            print('No unique identifier column found.')
