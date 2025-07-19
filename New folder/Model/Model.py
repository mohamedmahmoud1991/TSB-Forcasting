# models.py

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=1,
                 random_state=None, n_jobs=None, verbosity=1, enable_categorical=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.enable_categorical = enable_categorical

    def fit(self, X, y):
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            enable_categorical=self.enable_categorical
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
