
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import XGBRegressorWrapper  # <--- imported custom model

class LoadForecaster:
    def __init__(self):
        self.scalers = {}
        self.etr_models = {}
        self.xgb_models = {}
        self.meta_learners = {}
        self.feature_columns = None
        self.target_variable = 'actual_load_MW'
        self.feature_groups = {
            'text_features': [],
            'load_features': [],
            'temporal_features': [],
            'embeddings': []
        }

