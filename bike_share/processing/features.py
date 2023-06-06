from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from bike_share.config.core import config

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """Embarked column Imputer"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        wkday_null_idx = X[X['weekday'].isnull() == True].index
        X.loc[wkday_null_idx, 'weekday'] = X.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill_value=X[self.variables].mode()[0]
        # X[self.variables].fillna('Clear', inplace=True)
        # print("00000000000000000: ", [self.variables].unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)
        
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings)

        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self,factor=1.5):
        self.factor = factor
        
    def outlier_detector(self,X):
        X = X.copy()
        X = pd.to_numeric(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self,X):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self,X):
        
        X = X.copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        
        return X

        
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[[config.model_config.weekday_var]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        encoded_weekday = self.encoder.transform(X[[config.model_config.weekday_var]])
        enc_wkday_features = self.encoder.get_feature_names_out([config.model_config.weekday_var])
        X[enc_wkday_features] = encoded_weekday
        
        return X
    
class ColumnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        # return X.drop(self.columns,axis=1)
        X.drop(labels=self.columns, axis=1, inplace=True)
        # print(type(X))
        # print(X.info())
        # print(X)
        return X

    def fit(self, X, y=None):
        return self 