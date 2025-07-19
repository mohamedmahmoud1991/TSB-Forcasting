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

"""## Load & Weather Preparation"""

# Define paths
weather_path = '/content/drive/MyDrive/FullDataSet/weather.csv'
bbc_zip_path = '/content/drive/MyDrive/FullDataSet/bbc.zip'
load_data_dir = '/content/drive/MyDrive/FullDataSet/'  # Directory containing "UK 2021.csv"'  # Directory containing "UK 2021.csv"

# Step 1: Load and Aggregate Weather Data
#ITs every 3 hours Basis
weather_data = pd.read_csv(weather_path)
weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])
#weather_data['date'] = weather_data['date_time'].dt.date
weather_data = weather_data.drop(columns=['humidity','windspeedKmph'])
print('Weather Data Loaded', weather_data.shape)
print('Weather Data Loaded', weather_data.columns.tolist())
print(weather_data.head(10))

weather_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/WeatherData_Every_Three_Hours_From_2016_TO_2021.csv', index=False)

import pandas as pd

def preprocess_weather_data(weather_path):
    """
    Preprocess weather data by converting to 30-minute intervals and setting proper datetime index

    Parameters:
    weather_path (str): Path to weather CSV file

    Returns:
    pandas.DataFrame: Processed weather data with 30-minute intervals
    """
    # Read the weather data
    weather_data = pd.read_csv(weather_path)

    # Convert date_time to datetime and set as index
    weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])
    weather_data.set_index('date_time', inplace=True)

    # Sort index to ensure proper resampling
    weather_data.sort_index(inplace=True)

    # Resample to 30-minute intervals and interpolate
    weather_30min = weather_data.resample('30T').interpolate(method='linear')

    # Verify the conversion
    print("Original data points:", len(weather_data))
    print("After 30-min resampling:", len(weather_30min))
    print("\nSample of processed data:")
    print(weather_30min.head())

    return weather_30min

# Example usage
weather_path = '/content/drive/MyDrive/FullDataSet/DataPrepared/WeatherData_Every_Three_Hours_From_2016_TO_2021.csv'
processed_weather = preprocess_weather_data(weather_path)

# Save the processed data
output_path = '/content/drive/MyDrive/FullDataSet/DataPrepared/WeatherData_30Min_Processed.csv'
processed_weather.to_csv(output_path)

print("\nProcessed data saved to:", output_path)
print("\nData range:", processed_weather.index.min(), "to", processed_weather.index.max())

import pandas as pd
import numpy as np

# Define the years (removed duplicate 2018)
years = [2016, 2017, 2018, 2019, 2020, 2021]

# Function to parse custom date format
def parse_custom_date(date_str):
    start_date = date_str.split(' - ')[0]
    return pd.to_datetime(start_date, format="%d.%m.%Y %H:%M")

# Load and concatenate all files
load_data = pd.concat([
    pd.read_csv(f"{load_data_dir}/UK {year}.csv")
    for year in years
])

# 1. Basic Data Cleaning
print("Starting data cleaning...")
print(f"Initial shape: {load_data.shape}")

# Parse dates FIRST
print("\nParsing dates...")
load_data['datetime'] = load_data['Time'].apply(parse_custom_date)
load_data = load_data.set_index('datetime')
load_data = load_data.drop(columns=['Time'])

# Now we can check for duplicates
print("\nChecking for duplicates...")
duplicates = load_data.index.duplicated(keep='first').sum()  # Check index for duplicates
print(f"Found {duplicates} duplicate timestamps")
load_data = load_data[~load_data.index.duplicated(keep='first')]  # Remove duplicates based on index

# 2. Handle Missing Values
print("\nChecking for missing values...")
print(load_data.isnull().sum())

# Fill missing values with interpolation for load data
load_data['day_ahead_MW'] = load_data['Day-ahead Total Load Forecast [MW]'].interpolate(method='time')
load_data['actual_load_MW'] = load_data['Actual Total Load [MW]'].interpolate(method='time')
#load_data = load_data.drop(columns=[['Day-ahead Total Load Forecast [MW]','Actual Total Load [MW]']])

# 3. Remove Duplicates
print("\nChecking for duplicates...")
# 4. Handle Outliers
print("\nHandling outliers...")

def remove_outliers(df, column, n_std=4):
    """Remove outliers that are n standard deviations away from mean"""
    mean = df[column].mean()
    std = df[column].std()
    outliers = abs(df[column] - mean) > (n_std * std)
    print(f"Found {outliers.sum()} outliers in {column}")
    return outliers

# Identify outliers
day_ahead_outliers = remove_outliers(load_data, 'day_ahead_MW')
actual_outliers = remove_outliers(load_data, 'actual_load_MW')

# Replace outliers with rolling median
window_size = 24  # 24 hours
load_data.loc[day_ahead_outliers, 'day_ahead_MW'] = load_data['day_ahead_MW'].rolling(
    window=window_size, center=True, min_periods=1).median()
load_data.loc[actual_outliers, 'actual_load_MW'] = load_data['actual_load_MW'].rolling(
    window=window_size, center=True, min_periods=1).median()
# 5. Ensure chronological order and no gaps
print("\nChecking for time continuity...")
load_data = load_data.sort_values('datetime')
time_diff = load_data.index.diff()
gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
if not gaps.empty:
    print(f"Found {len(gaps)} gaps in time series")


# 7. Calculate quality metrics
load_data['forecast_error'] = load_data['actual_load_MW'] - load_data['day_ahead_MW']
load_data['forecast_error_pct'] = (load_data['forecast_error'] / load_data['actual_load_MW']) * 100

# 8. Ensure data types are correct
load_data['day_ahead_MW'] = load_data['day_ahead_MW'].astype(float)
load_data['actual_load_MW'] = load_data['actual_load_MW'].astype(float)

# Print summary statistics
print("\nSummary statistics after cleaning:")
print(load_data.describe())

# Print data quality report
print("\nData Quality Report:")
print(f"Total rows: {len(load_data)}")
print(f"Size: {(load_data.shape)}")
print(f"Date range: {load_data.index.min()} to {load_data.index.max()}")
print(f"Missing values: {load_data.isnull().sum().sum()}")
print(f"Mean absolute forecast error: {abs(load_data['forecast_error']).mean():.2f} MW")
print(f"Mean absolute percentage error: {abs(load_data['forecast_error_pct']).mean():.2f}%")

#print(load_data.columns.tolist())
#load_data=load_data.drop(columns=['forecast_error','forecast_error_pct'])
print('Load Data Loaded', load_data.shape)
print(load_data.head(3))
print(load_data.columns.tolist())
load_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Daily_30_Mins_Cleaned', index=False)

start_date = pd.to_datetime('2016-05-01 00:00:00')
end_date = pd.to_datetime('2021-07-31 21:00:00')

filtered_load_data = load_data[(load_data.index >= start_date) & (load_data.index <= end_date)]
filtered_load_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Daily_30_Mins_Cleaned_From_June2016_Till_May_2021', index=False)
print('Load Data Loaded', filtered_load_data.shape)
print(filtered_load_data.head(3))
print(filtered_load_data.columns.tolist())

import pandas as pd
import numpy as np

# Define the years (removed duplicate 2018)
years = [2016, 2017, 2018, 2019, 2020, 2021]


# Function to parse custom date format
def parse_custom_date(date_str):
    start_date = date_str.split(' - ')[0]
    return pd.to_datetime(start_date, format="%d.%m.%Y %H:%M")

# Load and concatenate all files
load_data = pd.concat([
    pd.read_csv(f"{load_data_dir}/UK {year}.csv")
    for year in years
])

# Rename 'Time' column to 'datetime' before processing
load_data = load_data.rename(columns={'Time': 'datetime'})

print('Load Data Loaded', load_data.shape)
print('Load Data Loaded', load_data.columns.tolist())
print(load_data.head(10))


# 1. Basic Data Cleaning
print("Starting data cleaning...")
print(f"Initial shape: {load_data.shape}")

# Parse dates
load_data['datetime'] = load_data['datetime'].apply(parse_custom_date)
load_data = load_data.set_index('datetime') # Set 'datetime' column as index


# Rename columns to standard format
load_data = load_data.rename(columns={
    'Day-ahead Total Load Forecast [MW]': 'day_ahead_MW',
    'Actual Total Load [MW]': 'actual_load_MW'
})

# 2. Handle Missing Values
print("\nChecking for missing values...")
print(load_data.isnull().sum())

# Fill missing values with interpolation for load data
load_data['day_ahead_MW'] = load_data['day_ahead_MW'].interpolate(method='time')
load_data['actual_load_MW'] = load_data['actual_load_MW'].interpolate(method='time')

# 3. Remove Duplicates
print("\nChecking for duplicates...")
duplicates = load_data.duplicated(subset=['datetime'], keep='first').sum()
print(f"Found {duplicates} duplicate timestamps")
load_data = load_data[~load_data.index.duplicated(keep='first')]

# 4. Handle Outliers
print("\nHandling outliers...")

def remove_outliers(df, column, n_std=4):
    """Remove outliers that are n standard deviations away from mean"""
    mean = df[column].mean()
    std = df[column].std()
    outliers = abs(df[column] - mean) > (n_std * std)
    print(f"Found {outliers.sum()} outliers in {column}")
    return outliers

# Identify outliers
day_ahead_outliers = remove_outliers(load_data, 'day_ahead_MW')
actual_outliers = remove_outliers(load_data, 'actual_load_MW')

# Replace outliers with rolling median
window_size = 24  # 24 hours
load_data.loc[day_ahead_outliers, 'day_ahead_MW'] = load_data['day_ahead_MW'].rolling(
    window=window_size, center=True, min_periods=1).median()
load_data.loc[actual_outliers, 'actual_load_MW'] = load_data['actual_load_MW'].rolling(
    window=window_size, center=True, min_periods=1).median()

# 5. Ensure chronological order and no gaps
print("\nChecking for time continuity...")
load_data = load_data.sort_values('datetime')
time_diff = load_data['datetime'].diff()
gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
if not gaps.empty:
    print(f"Found {len(gaps)} gaps in time series")

# 8. Ensure data types are correct
load_data['day_ahead_MW'] = load_data['day_ahead_MW'].astype(float)
load_data['actual_load_MW'] = load_data['actual_load_MW'].astype(float)

# Print summary statistics
print("\nSummary statistics after cleaning:")
print(load_data.describe())

# Print data quality report
print("\nData Quality Report:")
print(f"Total rows: {len(load_data)}")
print(f"Date range: {load_data['datetime'].min()} to {load_data['datetime'].max()}")
print(f"Missing values: {load_data.isnull().sum().sum()}")
print(f"Mean absolute forecast error: {abs(load_data['forecast_error']).mean():.2f} MW")
print(f"Mean absolute percentage error: {abs(load_data['forecast_error_pct']).mean():.2f}%")

# Save the processed data
load_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Daily_30_Mins_Cleaned', index=False)

import pandas as pd

def merge_load_weather_data(load_path, weather_path):
    """
    Merge load and weather data, resampling weather data to match load data frequency.

    Parameters:
    load_path (str): Path to load CSV file
    weather_path (str): Path to weather CSV file

    Returns:
    pandas.DataFrame: Merged dataset with load and interpolated temperature data
    """
    # Read load data
    load_df = pd.read_csv(load_path)

    # Read weather data
    weather_df = pd.read_csv(weather_path)

    # Merge the datasets
    merged_df = pd.merge(load_df,
                         weather_df,
                         left_index=True,
                         right_index=True,
                         how='inner')

    return merged_df

# Example usage
loadpath = '/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Daily_30_Mins_Cleaned_From_June2016_Till_May_2021'
weatherpath = '/content/drive/MyDrive/FullDataSet/DataPrepared/WeatherData_30Min_Processed.csv'
merged_data = merge_load_weather_data(loadpath, weatherpath)

# Display first few rows to verify the merge
print("\nFirst few rows of merged dataset:")
print(merged_data.head())

# Display basic statistics
print("\nDataset information:")
print(merged_data.info())

# Check for any remaining missing values
print("\nMissing values count:")
print(merged_data.isnull().sum())
merged_data.set_index('date_time', inplace=True)

print(merged_data.head(3))
merged_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_Cleaned_From_June2016_Till_May_2021', index=True)

merged_data=pd.read_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_Cleaned_From_June2016_Till_May_2021')
print(merged_data.shape)
print(merged_data.columns.to_list())
print(merged_data.index.name, merged_data.index.dtype)

!pip install holidays pandas numpy
import holidays
import pandas as pd

merged_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_Cleaned_From_June2016_Till_May_2021', index_col='date_time', parse_dates=True)

# --- The Fix ---
# Ensure 'date_time' is recognized as a DatetimeIndex
merged_data.index = pd.to_datetime(merged_data.index)
# --- End of Fix ---

uk_holidays = holidays.country_holidays('UK')  # Or your specific region
merged_data['holiday'] = merged_data.index.map(lambda x: 1 if x in uk_holidays else 0)

merged_data['hourofday'] = merged_data.index.hour  # Extract hour of day
merged_data['hourofday_sin'] = np.sin(2 * np.pi * merged_data['hourofday'] / 24)
merged_data['hourofday_cos'] = np.cos(2 * np.pi * merged_data['hourofday'] / 24)
merged_data['dayofweek'] = merged_data.index.dayofweek  # Monday=0, Sunday=6
merged_data['dayofmonth'] = merged_data.index.day
merged_data['weekofmonth'] = (merged_data.index.day - 1) // 7 + 1
merged_data['month'] = merged_data.index.month
merged_data['quarter'] = merged_data.index.quarter
merged_data['year'] = merged_data.index.year
merged_data['weekofyear'] = merged_data.index.isocalendar().week
uk_holidays = holidays.country_holidays('UK')  # Or your specific region
merged_data['holiday'] = merged_data.index.map(lambda x: 1 if x in uk_holidays else 0)
merged_data['dayofweek_sin'] = np.sin(2 * np.pi * merged_data['dayofweek'] / 7)
merged_data['dayofweek_cos'] = np.cos(2 * np.pi * merged_data['dayofweek'] / 7)
merged_data['month_sin'] = np.sin(2 * np.pi * merged_data['month'] / 12)
merged_data['month_cos'] = np.cos(2 * np.pi * merged_data['month'] / 12)
merged_data['is_business_day'] = merged_data['dayofweek'].apply(lambda x: 1 if x < 5 else 0)


# Add dayofyear feature
merged_data['dayofyear'] = merged_data.index.dayofyear


# --- The Fix ---
# Calculate lags for 8 half-hourly periods, previous day, week, month, and year
merged_data['actual_load_lag8'] = merged_data['actual_load_MW'].shift(8)  # Lag of 8 half-hours (4 hours)
merged_data['actual_load_lag_1day'] = merged_data['actual_load_MW'].shift(48)  # Lag of 1 day (48 half-hours)
merged_data['actual_load_lag_1week'] = merged_data['actual_load_MW'].shift(48 * 7)  # Lag of 1 week
merged_data['actual_load_lag_1month'] = merged_data['actual_load_MW'].shift(48 * 30)  # Approx. lag of 1 month
merged_data['actual_load_lag_1year'] = merged_data['actual_load_MW'].shift(48 * 365)  # Lag of 1 year

merged_data['dayofyear_sin'] = np.sin(2 * np.pi * merged_data['dayofyear'] / 365.25)
merged_data['dayofyear_cos'] = np.cos(2 * np.pi * merged_data['dayofyear'] / 365.25)
merged_data = merged_data.dropna()  # Drop rows with NaN values due to lag
# --- End of Fix ---

merged_data.to_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_From_June2016_Till_May_2021_Added_Analysis_Features')
print(merged_data.shape)
print(merged_data.columns.to_list())
#print(merged_data.describe)
print(merged_data.info())
print('theIndex is:',merged_data.index.name, merged_data.index.dtype)

"""## End of Prepartion For Load, Weather"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ShortTermLoadForecaster:
    def __init__(self, target_hours=2):
        self.target_hours = target_hours
        self.models = []
        self.feature_cols = None
        self.scaler = None

    def prepare_features_targets(self, df):
        """
        Prepare features and targets for short-term forecasting
        """
        # Create future target values
        targets = []
        for i in range(self.target_hours * 2):
            targets.append(df['actual_load_MW'].shift(-(i+1)))

        # Combine targets into a single dataframe
        y = pd.concat(targets, axis=1)
        y.columns = [f'actual_load_t{i+1}' for i in range(self.target_hours * 2)]

        # Prepare features
        if self.feature_cols is None:
            self.feature_cols = [col for col in df.columns
                               if col not in ['actual_load_MW', 'datetime']]

        X = df[self.feature_cols]

        # Remove rows with NaN targets
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        return X, y

    def evaluate_forecasts(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_pred) + np.abs(y_true)))

        return {
            'rmse': rmse,
            'mae': mae,
            'smape': smape
        }

    def train(self, X, y):
        """
        Train models for each forecast period
        """
        self.models = []

        for i in range(y.shape[1]):
            model = ExtraTreesRegressor(
                n_estimators=100,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y.iloc[:, i])
            self.models.append(model)

    def predict(self, X):
        """
        Generate predictions for all periods
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)

    def plot_feature_importance(self):
        """
        Plot feature importance for each forecast period
        """
        importance_df = pd.DataFrame()

        for i, model in enumerate(self.models):
            importance = model.feature_importances_
            period_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance,
                'period': f't{i+1}'
            })
            importance_df = pd.concat([importance_df, period_importance])

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='feature', y='importance', hue='period')
        plt.xticks(rotation=45)
        plt.title('Feature Importance by Forecast Period')
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, y_true, y_pred, start_date=None, end_date=None):
        """
        Plot actual vs predicted values
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i in range(y_true.shape[1]):
            ax = axes[i]
            ax.plot(y_true.iloc[:, i], label='Actual', alpha=0.7)
            ax.plot(y_pred[:, i], label='Predicted', alpha=0.7)
            ax.set_title(f'Period {i+1} (t+{(i+1)*30}min)')
            ax.legend()
            ax.grid(True)

            if start_date and end_date:
                ax.set_xlim(start_date, end_date)

        plt.tight_layout()
        plt.show()

def run_forecasting_pipeline(merged_df, plot_results=True):
    """
    Run complete forecasting pipeline with cross-validation
    """
    # Initialize forecaster
    forecaster = ShortTermLoadForecaster(target_hours=2)

    # Split train/test
    train_end = '2020-05-31'
    test_start = '2020-06-01'

    train_df = merged_df[merged_df.index <= train_end].copy()
    test_df = merged_df[merged_df.index >= test_start].copy()

    print("Data Split:")
    print(f"Training: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Testing: {test_df.index.min()} to {test_df.index.max()}")

    # Prepare data
    X_train, y_train = forecaster.prepare_features_targets(train_df)
    X_test, y_test = forecaster.prepare_features_targets(test_df)

    # Perform cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_metrics = []

    print("\nCross-validation Results:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        # Split data for this fold
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train and evaluate
        fold_forecaster = ShortTermLoadForecaster(target_hours=2)
        fold_forecaster.feature_cols = forecaster.feature_cols
        fold_forecaster.train(X_fold_train, y_fold_train)

        # Make predictions
        y_fold_pred = fold_forecaster.predict(X_fold_val)
        metrics = fold_forecaster.evaluate_forecasts(y_fold_val, y_fold_pred)
        cv_metrics.append(metrics)

        print(f"\nFold {fold}:")
        print(f"RMSE: {metrics['rmse']:.2f} MW")
        print(f"MAE: {metrics['mae']:.2f} MW")
        print(f"SMAPE: {metrics['smape']:.2f}%")

    # Train final model and evaluate on test set
    print("\nTraining final model...")
    forecaster.train(X_train, y_train)

    # Generate predictions
    y_pred_test = forecaster.predict(X_test)
    final_metrics = forecaster.evaluate_forecasts(y_test, y_pred_test)

    print("\nTest Set Results:")
    print(f"RMSE: {final_metrics['rmse']:.2f} MW")
    print(f"MAE: {final_metrics['mae']:.2f} MW")
    print(f"SMAPE: {final_metrics['smape']:.2f}%")

    if plot_results:
        print("\nPlotting feature importance...")
        forecaster.plot_feature_importance()

        print("\nPlotting predictions...")
        forecaster.plot_predictions(y_test, y_pred_test)

    return {
        'forecaster': forecaster,
        'cv_metrics': cv_metrics,
        'test_metrics': final_metrics,
        'test_predictions': y_pred_test,
        'test_actual': y_test
    }

# Run the pipeline
results = run_forecasting_pipeline(merged_data)

# Access the trained forecaster
forecaster = results['forecaster']

# Make new predictions
new_predictions = forecaster.predict(new_data)

# To make predictions for new data, it needs to have the same features as training data
# Let's check what features the model expects
print("Required features:", forecaster.feature_cols)

# Prepare new data example (should contain all required features)
# Let's take the last 48 periods (24 hours) from test data as an example
new_data = merged_data.iloc[-48:].copy()

# Ensure new_data has all required features
X_new = new_data[forecaster.feature_cols]

# Now make predictions
new_predictions = forecaster.predict(X_new)

# Convert predictions to a DataFrame with timestamps
prediction_df = pd.DataFrame(
    new_predictions,
    index=X_new.index,
    columns=[f'predicted_load_t{i+1}' for i in range(4)]  # 4 periods ahead
)

print("\nPredictions shape:", prediction_df.shape)
print("\nPredictions preview:")
print(prediction_df.head())