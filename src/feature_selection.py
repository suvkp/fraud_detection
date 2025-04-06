import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from itertools import combinations
from abc import abstractmethod, ABC

class FeatureSelectionBaseClass(ABC):
    @abstractmethod
    def select_features(self, data):
        pass

class RemoveHighMissingValues(FeatureSelectionBaseClass):
    """ Remove features with high percentage of missing values. """
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def select_features(self, data, target=None):
        missing_percentage = data.isnull().mean()
        selected_features = missing_percentage[missing_percentage < self.threshold].index
        return data[selected_features]
    
class PCAFeatureSelection(FeatureSelectionBaseClass):
    """ Select features using PCA. """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def select_features(self, data, target=None):
        pca = PCA(n_components=self.n_components if self.n_components else data.shape[1])
        X_pca = pca.fit_transform(data)
        return pd.DataFrame(X_pca)
    
class VarianceThresholdFeatureSelection(FeatureSelectionBaseClass):
    """ Select features based on variance threshold. """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def select_features(self, data, target=None):
        selector = VarianceThreshold(threshold=self.threshold)
        X_reduced = selector.fit_transform(data)
        return pd.DataFrame(X_reduced, columns=data.columns[selector.get_support()])
    
class CorrelationThresholdFeatureSelection(FeatureSelectionBaseClass):
    """ Select features based on correlation threshold. """
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def select_features(self, data, target=None):
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]
        return data.drop(columns=drop_columns)
    
class TargetCorrelationFeatureSelection(FeatureSelectionBaseClass):
    """ Select features based on correlation with the target variable. """
    def __init__(self, target, threshold=0.1):
        self.target = target
        self.threshold = threshold

    def select_features(self, data, target=None):
        correlations = data.corrwith(self.target).abs()
        selected_features = correlations[correlations > self.threshold].index
        return data[selected_features]
    
class ForwardSelectionFeatureSelection(FeatureSelectionBaseClass):
    """ Select features using forward selection method. """
    def __init__(self, estimator, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select)

    def select_features(self, data, target):
        self.selector.fit(data, target)
        selected_features = data.columns[self.selector.get_support()]
        return data[selected_features]
    
class FeatureSelection:
    def __init__(self, feature_selector: FeatureSelectionBaseClass) -> None:
        """
        Args
            method: Feature selection method to apply. Options include:
                - 'missing_threshold': Remove features with high percentage of missing values.
                - 'pca': Apply PCA for dimensionality reduction.
                - 'variance_threshold': Remove features with low variance.
                - 'correlation_threshold': Remove features with high correlation.
                - 'target_correlation': Select features based on correlation with target variable.
                - 'forward_selection': Select features using forward selection method.
            kwargs: Additional parameters for the selected method.
        """
        self.feature_selector = feature_selector

    def select_features(self, data, target=None):
        """
        select_features method to apply the selected feature selection method.
        Args: 
            data: DataFrame containing the features
            target: Series or DataFrame containing the target variable (if applicable)
        Returns:
            DataFrame with selected features
        """
        try:
            return self.feature_selector.select_features(data, target)
        except Exception as e:
            print(f"Error in feature selection: {e}")
            raise e
