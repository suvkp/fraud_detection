import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from itertools import combinations

class FeatureSelectionExperiment:
    def __init__(self, method, missing_threshold=0.3, variance_threshold=0.01, 
                 correlation_threshold=0.9, target_correlation_threshold=0.1, n_pca_components=None):
        """
        Initialize the feature selection experiment.
        
        :param method: Feature selection method to apply.
        :param missing_threshold: Threshold for removing features with missing values.
        :param variance_threshold: Minimum variance required for a feature to be kept.
        :param correlation_threshold: Threshold for removing one of two highly correlated features.
        :param target_correlation_threshold: Minimum correlation with target to keep a feature.
        :param n_pca_components: Number of PCA components to keep (if using PCA).
        """
        self.method = method
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_correlation_threshold = target_correlation_threshold
        self.n_pca_components = n_pca_components
    
    def remove_high_missing_features(self, X):
        missing_percentage = X.isnull().mean()
        selected_features = missing_percentage[missing_percentage < self.missing_threshold].index
        return X[selected_features]
    
    def apply_pca(self, X):
        pca = PCA(n_components=self.n_pca_components if self.n_pca_components else X.shape[1])
        X_pca = pca.fit_transform(X)
        return pd.DataFrame(X_pca)
    
    def apply_variance_threshold(self, X):
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_reduced = selector.fit_transform(X)
        return pd.DataFrame(X_reduced, columns=X.columns[selector.get_support()])
    
    def remove_highly_correlated_features(self, X):
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]
        return X.drop(columns=drop_columns)
    
    def select_features_by_target_correlation(self, X, y):
        correlations = X.corrwith(y).abs()
        selected_features = correlations[correlations > self.target_correlation_threshold].index
        return X[selected_features]
    
    def forward_selection(self, X, y):
        selected_features = []
        remaining_features = list(X.columns)
        best_score = -np.inf

        while remaining_features:
            scores = []
            for feature in remaining_features:
                model = LinearRegression()
                model.fit(X[selected_features + [feature]], y)
                score = model.score(X[selected_features + [feature]], y)
                scores.append((feature, score))
            
            best_feature, best_feature_score = max(scores, key=lambda x: x[1])
            if best_feature_score > best_score:
                best_score = best_feature_score
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break
        
        return X[selected_features]
    
    def run(self, X, y=None):
        """
        Run the selected feature selection method on the dataset.
        
        :param X: Feature matrix.
        :param y: Target variable (required for some methods).
        :return: Transformed dataset with selected features.
        """
        if self.method == 'missing_threshold':
            return self.remove_high_missing_features(X)
        elif self.method == 'pca':
            return self.apply_pca(X)
        elif self.method == 'variance_threshold':
            return self.apply_variance_threshold(X)
        elif self.method == 'correlation_threshold':
            return self.remove_highly_correlated_features(X)
        elif self.method == 'target_correlation':
            if y is None:
                raise ValueError("Target variable is required for feature selection based on correlation with target.")
            return self.select_features_by_target_correlation(X, y)
        elif self.method == 'forward_selection':
            if y is None:
                raise ValueError("Target variable is required for forward selection.")
            return self.forward_selection(X, y)
        else:
            raise ValueError("Invalid feature selection method provided.")
