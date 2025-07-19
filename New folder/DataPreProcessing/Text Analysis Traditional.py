
# Install necessary libraries
!pip install nltk textblob scikit-learn statsmodels lime econml

# Download necessary NLTK data files
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import zipfile
import pandas as pd
import os
import json
import glob
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import grangercausalitytests
from econml.dml import LinearDML
import numpy as np
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""# **SBert Processing**"""

news_df = pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/NewsBeforePreProcessing.csv')
print(f'\n news_df Description:')
print(f'news_df features: {len(news_df)}')
print(f'news_df shape: {news_df.shape}')
print(f'news_df Column: {news_df.columns.to_list()}')
news_df = news_df.fillna("")


# Assuming your DataFrame is called news_df
news_df['combined_col'] = news_df['title'] + ' ' + news_df['content'] + ' ' + news_df['description'] + ' ' + news_df['section']

# Keep only the 'date' and 'text' columns
news_df = news_df[['date', 'combined_col']]

print(f'\n news_df Description:')
print(f'news_df features: {len(news_df)}')
print(f'news_df shape: {news_df.shape}')
print(f'news_df Column: {news_df.columns.to_list()}')

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
!pip install huggingface_hub
from huggingface_hub import login

login(token="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") # Replace YOUR_HF_TOKEN with your actual token

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm
import gc  # For garbage collection

def add_sbert_embeddings_optimized(news_df, batch_size=128):
    """
    Optimized SBERT embedding generation for Colab
    """
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize SBERT model
    model = SentenceTransformer('all-mpnet-base-v2')
    model = model.to(device)

    # Process in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 articles at a time
    embeddings_list = []

    for chunk_start in tqdm(range(0, len(news_df), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(news_df))
        # Get the chunk of data
        chunk_data = news_df['combined_col'].iloc[chunk_start:chunk_end]
        # Convert each row of the chunk to a list and extend the embeddings_list
        chunk_texts = chunk_data.values.tolist() #The values attribute of the DataFrame gives you a NumPy array, and then you can use tolist() to convert it into a list of lists.

        # Generate embeddings for the chunk
        with torch.no_grad():  # Disable gradient calculation
            chunk_embeddings = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

        embeddings_list.append(chunk_embeddings)

        # Clear CUDA cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

    # Concatenate all embeddings
    embeddings = np.vstack(embeddings_list)

    # Add essential embeddings features only (to reduce memory usage)
    # Add mean embedding for each dimension
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f'sbert_dim_{i}' for i in range(embeddings.shape[1])]
    )

    # Calculate aggregate features
    news_df['sbert_mean'] = emb_df.mean(axis=1)
    news_df['sbert_std'] = emb_df.std(axis=1)

    # Add only the first 10 dimensions as individual features
    for i in range(min(10, embeddings.shape[1])):
        news_df[f'sbert_dim_{i}'] = embeddings[:, i]

    return news_df

# Usage example:
news_df = add_sbert_embeddings_optimized(news_df)
news_df.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`9_News_After_SBert(Combined_Features_For_News)_23_12_2024_Before_Aggregating.csv', index=False)

print(f'\n news_df Description After SBert:')
print(f'news_df features: {len(news_df)}')
print(f'news_df shape: {news_df.shape}')
print(f'news_df Column: {news_df.columns.to_list()}')

def aggregate_daily_embeddings_optimized(news_df):
    """
    Memory-efficient daily aggregation with unique dates,
    calculating std before dropping duplicates.
    """
    # Get embedding columns
    embedding_cols = news_df.select_dtypes(exclude=['object']).columns.tolist()

    # Define focused aggregations
    agg_dict = {
        col: ['mean', 'std'] for col in embedding_cols
    }

    # Calculate daily aggregations before dropping duplicates
    daily_embeddings = news_df.groupby('date')[embedding_cols].agg(agg_dict)

    # Flatten column names
    daily_embeddings.columns = [f'{col[0]}_{col[1]}' for col in daily_embeddings.columns]

    # Ensure unique dates after aggregation (if needed)
    # daily_embeddings = daily_embeddings[~daily_embeddings.index.duplicated(keep='first')]

    return daily_embeddings

news_df=pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`9_News_After_SBert(Combined_Features_For_News)_23_12_2024_Before_Aggregating.csv', index_col=False)
daily_embeddings=aggregate_daily_embeddings_optimized(news_df)
daily_embeddings.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`8_News_After_SBertSBert(Combined_Features_For_News)_23_12_2024_After_Aggregating(DailyEmbeddings)_Date_is_the_index.csv')
print(f'\n news_df Description After SBert Aggregated to Daily Embeddings:')
print(daily_embeddings.shape)
print(daily_embeddings.index.name)
print(daily_embeddings.columns.to_list())
print(daily_embeddings.head(1))

"""# **FEATURE Combination**
Load Weather & LoadData & Calender & Lag & Holidays Combined

"""

# Merge Data
news_df=pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`8_News_After_SBertSBert(Combined_Features_For_News)_23_12_2024_After_Aggregating(DailyEmbeddings)_Date_is_the_index.csv',)
merged_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/FinalData_After_Embeddings_Exp1_Code(load, Time2Vec)_21_12_2024.csv')

print(news_df.shape)
print(news_df.index.name)
print(news_df.columns.to_list())
print(news_df.head(2))

print(merged_data.shape)
print(merged_data.index.name)
print(merged_data.columns.to_list())
print(merged_data.head(2))

# Convert date columns to datetime
news_df['date'] = pd.to_datetime(news_df['date'])  # Assuming 'date' is the index

# Convert date_time column to datetime and extract date component
merged_data['date_time'] = pd.to_datetime(merged_data['date_time'])
merged_data['date'] = merged_data['date_time'].dt.date

# Convert date column back to datetime for merging
merged_data['date'] = pd.to_datetime(merged_data['date'])

print(news_df.shape)
print(news_df.index.name)
print(news_df.columns.to_list())
print(news_df.head(2))

print(merged_data.shape)
print(merged_data.index.name)
print(merged_data.columns.to_list())
print(merged_data.head(2))

final_data = pd.merge(news_df, merged_data, on='date', how='left')
# Merge the dataframes
print(f'\n FinalData Description all features:')
print(f'FinalData features: {len(final_data)}')
print(f'FinalData shape: {final_data.shape}')
print(f'FinalData Column: {final_data.columns.to_list()}')

# Assuming your DataFrame is called 'final_data'
columns_to_keep = ['date_time']  # Start with 'date_time'

# Add columns with numerical data types
for column in final_data.columns:
    if pd.api.types.is_numeric_dtype(final_data[column]):  # Check for numerical type
        columns_to_keep.append(column)

# Select only the desired columns
final_data = final_data[columns_to_keep]

# Merge the dataframes
print(f'\n FinalData Description Numerics only:')
print(f'FinalData features: {len(final_data)}')
print(f'news_df shape: {news_df.shape}')
print(f'merged_data shape: {merged_data.shape}')
print(f'FinalData shape: {final_data.shape}')
print(f'FinalData Column: {final_data.columns.to_list()}')

final_data.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`9_(Adding_Time2VEC+SBert_(Title+Content+Description+Section)_CombinedColumn_23_12_2024.csv', index=False)

"""# **Causality**"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

class SHAPFeatureSelector:
    def __init__(self, target_col: str = 'actual_load_MW',keep_cols=['date_time']):
        self.target_col = target_col
        self.keep_cols = keep_cols

    def select_features(self, data: pd.DataFrame, n_top_features: int = 100) -> pd.DataFrame:
        """
        Perform SHAP-based feature selection.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing features and the target column.
            n_top_features (int): Number of top features to select based on SHAP importance.

        Returns:
            pd.DataFrame: DataFrame containing only the selected features and the target column.
        """
        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(data[self.target_col]):
            raise ValueError(f"Target column '{self.target_col}' must be numeric.")

        # Filter only numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data[[col for col in numeric_data.columns if col not in self.keep_cols]]
        if self.target_col not in numeric_data.columns:
            raise ValueError(f"Target column '{self.target_col}' is missing from numeric data.")

        # Split features and target
        X = numeric_data.drop(columns=[self.target_col])
        y = numeric_data[self.target_col]

        X = X.dropna()  # Drop rows with NaN in features
        y = y[X.index]  # Ensure y has the same index as X after dropping rows


        # Train a RandomForestRegressor
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X, y)
        # Drop rows with NaN values in either X or y


        # Calculate SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Calculate mean absolute SHAP values for each feature
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': shap_importance
        }).sort_values(by='Importance', ascending=False)

        # Select top N features
        top_features = feature_importance.head(n_top_features)['Feature'].tolist()
        print("Top selected features:", top_features)

        # Return the DataFrame with selected features and the target column
        selected_data = data[[*top_features, *self.keep_cols, self.target_col]]
        return selected_data, feature_importance

    def visualize_importance(self, feature_importance: pd.DataFrame, top_n=10):  # Add top_n parameter
        """Visualize feature importance."""

        # Select top N features
        top_features = feature_importance.head(top_n)  # Select top_n features

        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')  # Plot only top_n
        plt.xlabel('SHAP Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance based on SHAP')  # Update title
        plt.gca().invert_yaxis()
        plt.show()


def main():
    print("Loading prepared data...")
    # Load the dataset
    final_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`9_(Adding_Time2VEC+SBert_(Title+Content+Description+Section)_CombinedColumn_23_12_2024.csv')  # Update with the actual path if needed

    print("Initial DataFrame shape:", final_data.shape)

    # Select 10,000 random rows for SHAP computations
    shap_sample = final_data.sample(n=10000, random_state=42)  # Sampling for SHAP computation
    print("SHAP sample DataFrame shape:", shap_sample.shape)

    # Initialize SHAP feature selector
    shap_selector = SHAPFeatureSelector(target_col='actual_load_MW')

    # Perform feature selection using the SHAP sample
    selected_sample_data, feature_importance = shap_selector.select_features(shap_sample, n_top_features=50)

    # Use the selected features on the entire dataset
    selected_features = feature_importance.head(50)['Feature'].tolist()  # Get top 50 feature names
    final_selected_data = final_data[[*selected_features, *shap_selector.keep_cols, shap_selector.target_col]]

    print("\nFeature selection completed.")
    print("Final DataFrame shape with selected features:", final_selected_data.shape)

    # Save selected features to a new CSV
    final_selected_data.to_csv('/content/drive/MyDrive/FullDataSet/Exp16_Output/selected_features.csv', index=False)
    print("Selected features saved to '/content/drive/MyDrive/FullDataSet/Exp16_Output/selected_features.csv'")

    # Visualize SHAP importance
    shap_selector.visualize_importance(feature_importance)

    # Display selected feature names
    print("\nRemaining features:", final_selected_data.columns.tolist())

if __name__ == "__main__":
    main()

final_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/Exp16_Output/selected_features.csv')
#final_data1 = pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`9_(Adding_Time2VEC+SBert_(Title+Content+Description+Section)_CombinedColumn_23_12_2024.csv')  # Update with the actual path if needed

print(f'\n news_df Description:')
print(f'news_df features: {len(final_data)}')
print(f'news_df shape: {final_data.shape}')
print(f'news_df Column: {final_data.columns.to_list()}')
# Assuming your DataFrame is named 'df'
if 'date_time' in final_data.columns:
  print("Column 'column_name' exists in the DataFrame.")
else:
  print("Column 'column_name' does not exist in the DataFrame.")

"""## **Train ETR**

Train ExtraTreesRegressor : Target: Next 2 hours (4 half-hour periods)

ExtraTrees Regression:

Combine selected features

Grid search for parameters

5-fold cross-validation (Train siz on first 4 years till End of April2020, Test size from May2020 till end of data)

Focus on 0h and 1h prediction on actual_koad_MW

Metrics calculation for 0h-1h:

RMSE

MAE

SMAPE

Focus on 2-hour ahead prediction
"""

import pandas as pd
import os
import json
import glob
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit # Import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import gc
from sklearn.linear_model import LinearRegression # Import LinearRegression

!pip install numpy pandas matplotlib xgboost scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import gc

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

    def train_models(self, X_train, y_train):
        etr_param_grid = {
            'n_estimators': [500],
            'max_features': ['sqrt'],
            'max_depth': [30, None],
            'min_samples_split': [5],
            'min_samples_leaf': [2],
            'bootstrap': [True],
            'max_samples': [0.8],
            'min_impurity_decrease': [0.0]
        }

        xgb_param_grid = {
            'n_estimators': [500],
            'max_depth': [6, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0.01, 0.1],
            'reg_lambda': [0.01, 0.1]
        }

        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X_train) * 0.2))

        for i in range(1, 5):
            print(f"\nTraining models for t{i} prediction horizon...")
            gc.collect()

            # Train ExtraTrees
            etr_base_model = ExtraTreesRegressor(random_state=42, n_jobs=2, criterion='squared_error', warm_start=True, verbose=1)
            etr_grid_search = GridSearchCV(etr_base_model, etr_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1, verbose=2, return_train_score=True)
            etr_grid_search.fit(X_train, y_train[f'target_t{i}'])
            self.etr_models[f't{i}'] = etr_grid_search.best_estimator_

            # Train XGBoost with wrapper
            xgb_base_model = XGBRegressorWrapper()

            xgb_grid_search = GridSearchCV(xgb_base_model, xgb_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1, verbose=2, return_train_score=True)
            xgb_grid_search.fit(X_train, y_train[f'target_t{i}'])
            self.xgb_models[f't{i}'] = xgb_grid_search.best_estimator_

            print(f"\nBest ETR parameters for t{i}:")
            print(etr_grid_search.best_params_)
            print(f"\nBest XGBoost parameters for t{i}:")
            print(xgb_grid_search.best_params_)

            gc.collect()

    def train_meta_learner(self, X_val, y_val):
        for i in range(1, 5):
            horizon = f't{i}'
            etr_preds = self.etr_models[horizon].predict(X_val)
            xgb_preds = self.xgb_models[horizon].model_.predict(X_val)  # Note: using .model for XGBoost wrapper

            meta_features = np.column_stack((etr_preds, xgb_preds))
            meta_learner = LinearRegression()
            meta_learner.fit(meta_features, y_val[f'target_t{i}'])
            self.meta_learners[horizon] = meta_learner

    def prepare_data(self, final_data):
        """
        Prepare data for forecasting
        """
        # Convert date_time to datetime if it's not already
        final_data['date_time'] = pd.to_datetime(final_data['date_time'])
        final_data = final_data.set_index('date_time')

        # Sort by date_time
        final_data = final_data.sort_index()

        # Create target variables for next 4 half-hour periods
        for i in range(1, 5):
            final_data[f'target_t{i}'] = final_data['actual_load_MW'].shift(-i)

        # Drop rows with NaN in targets
        final_data = final_data.dropna()

        # Select features (all numeric columns except targets)
        self.feature_columns = [col for col in final_data.columns
                              if not col.startswith('target_') and
                              final_data[col].dtype in ['int64', 'float64']]

        print(f'\nPrepare Data Description:')
        print(f'Number of samples: {len(final_data)}')
        print(f'Data shape: {final_data.shape}')
        print(f'Columns: {final_data.columns.to_list()}')
        return final_data

    def create_train_test_split(self, data):
        """Create train-test split based on time"""
        split_date = '2020-04-30'
        train_data = data[data.index <= split_date]
        test_data = data[data.index > split_date]

        print(f'\nTrain Data Description:')
        print(f'Train samples: {len(train_data)}')
        print(f'Train shape: {train_data.shape}')

        print(f'\nTest Data Description:')
        print(f'Test samples: {len(test_data)}')
        print(f'Test shape: {test_data.shape}')

        return train_data, test_data

    def scale_features(self, train_data, test_data):
        """Scale features using StandardScaler"""
        X_train = train_data[self.feature_columns].copy()
        X_test = test_data[self.feature_columns].copy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['features'] = scaler

        return (pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index),
                pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index))

    def predict_and_evaluate(self, X_test, y_test):
        results = {}
        predictions = {}

        for i in range(1, 5):
            horizon = f't{i}'
            etr_preds = self.etr_models[horizon].predict(X_test)
            xgb_preds = self.xgb_models[horizon].model_.predict(X_test)  # Note: using .model for XGBoost wrapper

            meta_features = np.column_stack((etr_preds, xgb_preds))
            predictions[horizon] = self.meta_learners[horizon].predict(meta_features)

            metrics = self.calculate_metrics(y_test[f'target_t{i}'], predictions[horizon])
            results[horizon] = metrics

            print(f"\nMetrics for {horizon} ({i/2:.1f} hour ahead):")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

        return predictions, results

    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Calculate SMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred))
        smape = 200 * np.mean(np.abs(y_true - y_pred) / denominator)

        return {
            'RMSE (MW)': rmse,
            'MAE (MW)': mae,
            'SMAPE (%)': smape
        }

    def plot_predictions(self, y_test, predictions, start_date, end_date):
        """Plot actual vs predicted load for a specified date range."""
        # Filter data for the specified date range
        mask = (y_test.index >= start_date) & (y_test.index <= end_date)
        y_test_filtered = y_test[mask]
        predictions_filtered = {k: v[mask] for k, v in predictions.items()}

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual load
        ax.plot(y_test_filtered.index, y_test_filtered['target_t1'], label='Actual Load', color='blue')

        # Plot predictions for each horizon
        for i, horizon in enumerate(['t1', 't2', 't3', 't4']):
            ax.plot(y_test_filtered.index, predictions_filtered[horizon],
                    label=f'Predicted Load (t{i+1})', linestyle='--', color=f'C{i+1}')

        # Set title and labels
        ax.set_title('Actual vs Predicted Load')
        ax.set_xlabel('Date')
        ax.set_ylabel('Load (MW)')
        ax.legend()

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Show plot
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = LoadForecaster()

    # Load and prepare data
    final_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/Exp16_Output/selected_features.csv')

    processed_data = forecaster.prepare_data(final_data)

    # Split data
    train_data, test_data = forecaster.create_train_test_split(processed_data)

    # Further split train data for meta-learner
    split_date = '2019-12-31'
    train_data_base = train_data[train_data.index <= split_date]
    train_data_meta = train_data[train_data.index > split_date]

    # Scale features
    X_train_base_scaled, X_train_meta_scaled = forecaster.scale_features(train_data_base, train_data_meta)
    _, X_test_scaled = forecaster.scale_features(train_data, test_data)

    # Prepare target variables
    y_train_base = train_data_base[[f'target_t{i}' for i in range(1, 5)]]
    y_train_meta = train_data_meta[[f'target_t{i}' for i in range(1, 5)]]
    y_test = test_data[[f'target_t{i}' for i in range(1, 5)]]

    # Train models
    forecaster.train_models(X_train_base_scaled, y_train_base)

    # Train meta-learner
    forecaster.train_meta_learner(X_train_meta_scaled, y_train_meta)

    # Make predictions and evaluate
    predictions, results = forecaster.predict_and_evaluate(X_test_scaled, y_test)

    # Print summary
    print("\nSummary of Predictions and Metrics:")
    for horizon, metrics in results.items():
        print(f"\nHorizon: {horizon}")
        print(f"  Mean Prediction: {np.mean(predictions[horizon]):.4f}")
        print(f"  Std of Predictions: {np.std(predictions[horizon]):.4f}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Plot results for a week
    forecaster.plot_predictions(y_test, predictions, '2020-05-01', '2020-05-07')