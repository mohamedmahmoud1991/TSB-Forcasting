
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.feature_columns = None
        self.target_variable = 'actual_load_MW'  # Explicitly define target variable

    def forward(self, x):
        return self.encoder(x)

def create_sequence_data(data, sequence_length=48):
    """Create sequences for embedding generation with proper shape handling"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def generate_embeddings(data, numeric_cols, sequence_length=48, embedding_dim=64):
    """Generate embeddings for time series data with proper shape handling"""
    sequences = []
    for col in numeric_cols:
        seq = create_sequence_data(data[col].values, sequence_length)
        sequences.append(seq)

    combined_sequences = np.stack(sequences, axis=2)
    sequences_tensor = torch.FloatTensor(combined_sequences)

    model = TimeSeriesEmbedding(input_dim=len(numeric_cols), embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        reshaped_input = sequences_tensor.reshape(-1, len(numeric_cols))
        embeddings = model(reshaped_input)
        loss = criterion(embeddings[:100], embeddings[1:101])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embeddings = model(sequences_tensor.reshape(-1, len(numeric_cols)))
        final_embeddings = final_embeddings.reshape(-1, sequence_length, embedding_dim)[:, -1, :]

    print(f'\n final_embeddings Description:')
    print(f'final_embeddings features: {len(final_embeddings)}')
    print(f'final_embeddings shape: {final_embeddings.shape}')
    #print(f'final_embeddings Column: {final_embeddings.columns.to_list()}')


    return final_embeddings.numpy()

def prepare_data_with_embeddings(merged_data):
    """Prepare data with embeddings ensuring proper index alignment"""
    embedding_cols = [
        'day_ahead_MW', 'actual_load_MW', 'tempC',
        'actual_load_lag8', 'actual_load_lag_1day',
        'actual_load_lag_1week', 'actual_load_lag_1month',
        'actual_load_lag_1year'
    ]

    print(f"Generating embeddings for features: {embedding_cols}")

    embeddings = generate_embeddings(
        merged_data,
        numeric_cols=embedding_cols,
        sequence_length=48,
        embedding_dim=64
    )

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f'embed_{i}' for i in range(embeddings.shape[1])],
        index=merged_data.index[47:]
    )

    final_data = pd.concat([
        merged_data.iloc[47:],
        embedding_df
    ], axis=1)

    # Create target variables for next 4 half-hour periods
    for i in range(1, 5):
        final_data[f'target_t{i}'] = final_data['actual_load_MW'].shift(-i)

    # Drop rows with NaN in targets
    final_data = final_data.dropna()

    print(f"Original data shape: {merged_data.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Final data shape: {final_data.shape}")

    return final_data

def evaluate_forecasts(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    return {'rmse': rmse, 'mae': mae, 'smape': smape}

def train_and_evaluate_model_cv(data, n_splits=5):
    """Train and evaluate the ETR model with 5-fold cross validation"""
    print(f'\n final_data inside the function:')
    print(f'\n final_data_with_embedding features: {len(data)}')
    print(f'\n final_data_with_embedding shape: {data.shape}')
    print(f'\n final_data_with_embedding Column: {data.columns.to_list()}')

    split_date = '2020-04-30'
    train_data = data[data.index <= split_date]
    test_data = data[data.index > split_date]
    print(f'\n train_data Description:')
    print(f'train_data features: {len(train_data)}')
    print(f'train_data shape: {train_data.shape}')
    print(f'train_data Column: {train_data.columns.to_list()}')

    print(f'\n test_data Description:')
    print(f'test_data features: {len(test_data)}')
    print(f'test_data shape: {test_data.shape}')
    print(f'test_data Column: {test_data.columns.to_list()}')

    X_train = train_data[self.feature_columns].copy()
    X_test = test_data[self.feature_columns].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare features and target
    X = data.drop('actual_load_MW', axis=1)
    y = data['actual_load_MW']

    print(f'\n x inside the function:')
    print(f'\n x features: {len(X)}')
    print(f'\n x shape: {X.shape}')
    print(f'\n x Column: {X.columns.to_list()}')

    print(f'\n y inside the function:')
    print(f'\n y features: {len(y)}')
    print(f'\n y shape: {y.shape}')
    print(f'\n y Column: {y.values}')
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize metrics storage
    cv_metrics = []

    # Initialize scaler
    scaler = StandardScaler()

    # Initialize model parameters
    model_params = {
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }


    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining fold {fold}...")
        print(f'\n y inside the function:')
        print(f'\n TrainingTest features: {len(train_idx),len(test_idx)}')
        print(f'\n TrainingTest shape: {(train_idx.shape), (test_idx.shape)}')
        #print(f'\n TrainingTest Column: {(train_idx.columns.to_list()), (test_idx.columns.to_list())}')

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print(f"\nTraining fold {fold}...")
        etr = ExtraTreesRegressor(**model_params)
        etr.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = etr.predict(X_test_scaled)

        # Calculate metrics
        fold_metrics = evaluate_forecasts(y_test, y_pred)
        cv_metrics.append(fold_metrics)

        print(f"Fold {fold} Results:")
        print(f"RMSE: {fold_metrics['rmse']:.2f} MW")
        print(f"MAE: {fold_metrics['mae']:.2f} MW")
        print(f"SMAPE: {fold_metrics['smape']:.2f}%")


    for i in range(1, 5):
        grid_search = GridSearchCV(
            ExtraTreesRegressor(random_state=42),
            model_params,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Train model for each horizon
        grid_search.fit(X_train, y_train[f'target_t{i}'])
        self.models[f't{i}'] = grid_search.best_estimator_

        print(f"\nBest parameters for t{i}:")
        print(grid_search.best_params_)


    print("\nPerforming 5-fold cross validation...")

    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in cv_metrics])
        for metric in ['rmse', 'mae', 'smape']
    }

    print("\nAverage Cross-Validation Results:")
    print(f"Average RMSE: {avg_metrics['rmse']:.2f} MW")
    print(f"Average MAE: {avg_metrics['mae']:.2f} MW")
    print(f"Average SMAPE: {avg_metrics['smape']:.2f}%")

    return cv_metrics, avg_metrics

def plot_cv_results(cv_metrics):
    """Plot cross-validation results"""
    metrics_df = pd.DataFrame(cv_metrics)

    plt.figure(figsize=(12, 6))
    metrics_df.boxplot()
    plt.title('Cross-Validation Metrics Distribution')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def main():
    # Load data
    print("Loading data...")
    merged_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_From_June2016_Till_May_2021_Added_Analysis_Features',
                             index_col='date_time',
                             parse_dates=True)

    print(f'\n merged_data Description:')
    print(f'merged_data features: {len(merged_data)}')
    print(f'merged_data shape: {merged_data.shape}')
    print(f'merged_data Column: {merged_data.columns.to_list()}')

    # Prepare data with embeddings
    print("\nPreparing data with embeddings...")
    final_data = prepare_data_with_embeddings(merged_data)

    print(f'\n final_data_with_embedding Description:')
    print(f'final_data_with_embedding features: {len(final_data)}')
    print(f'final_data_with_embedding shape: {final_data.shape}')
    print(f'final_data_with_embedding Column: {final_data.columns.to_list()}')

    final_data.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/FinalData_After_Embeddings_Exp1_Code(load, Time2Vec).csv')
    # Perform cross-validation
    cv_metrics, avg_metrics = train_and_evaluate_model_cv(final_data)

    # Plot results
    plot_cv_results(cv_metrics)

    return cv_metrics, avg_metrics

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def create_sequence_data(data, sequence_length=48):
    """Create sequences for embedding generation."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def generate_embeddings(data, numeric_cols, sequence_length=48, embedding_dim=64):
    """Generate embeddings for selected numeric features."""
    sequences = []
    for col in numeric_cols:
        seq = create_sequence_data(data[col].values, sequence_length)
        sequences.append(seq)

    combined_sequences = np.stack(sequences, axis=2)
    sequences_tensor = torch.FloatTensor(combined_sequences)

    model = TimeSeriesEmbedding(input_dim=len(numeric_cols), embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        reshaped_input = sequences_tensor.reshape(-1, len(numeric_cols))
        embeddings = model(reshaped_input)
        loss = criterion(embeddings[:-1], embeddings[1:])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embeddings = model(sequences_tensor.reshape(-1, len(numeric_cols)))
        final_embeddings = final_embeddings.reshape(-1, sequence_length, embedding_dim)[:, -1, :]

    print(f"Generated embeddings with shape: {final_embeddings.shape}")
    return final_embeddings.numpy()

def prepare_data_with_embeddings(merged_data):
    """Prepare dataset with time series embeddings."""
    embedding_cols = ['day_ahead_MW', 'actual_load_MW', 'tempC',
                      'actual_load_lag8', 'actual_load_lag_1day',
                      'actual_load_lag_1week', 'actual_load_lag_1month',
                      'actual_load_lag_1year']

    print(f"Generating embeddings for: {embedding_cols}")

    embeddings = generate_embeddings(
        merged_data, numeric_cols=embedding_cols, sequence_length=48, embedding_dim=64
    )

    embedding_df = pd.DataFrame(
        embeddings, columns=[f'embed_{i}' for i in range(embeddings.shape[1])],
        index=merged_data.index[47:]
    )

    final_data = pd.concat([merged_data.iloc[47:], embedding_df], axis=1)

    for i in range(1, 5):
        final_data[f'target_t{i}'] = final_data['actual_load_MW'].shift(-i)

    final_data = final_data.dropna()
    print(f"Final dataset shape after embeddings: {final_data.shape}")
    return final_data

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    return {'RMSE': rmse, 'MAE': mae, 'SMAPE': smape}

def train_and_evaluate_model(final_data):
    """Train ExtraTreesRegressor and evaluate performance."""
    split_date = '2020-04-30'
    train_data = final_data[final_data.index <= split_date]
    test_data = final_data[final_data.index > split_date]

    X_train = train_data.drop(columns=[f'target_t{i}' for i in range(1, 5)])
    X_test = test_data.drop(columns=[f'target_t{i}' for i in range(1, 5)])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = train_data[[f'target_t{i}' for i in range(1, 5)]]
    y_test = test_data[[f'target_t{i}' for i in range(1, 5)]]

    model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    metrics = {}
    predictions = {}

    for i in range(1, 5):
        print(f"Training model for t{i}...")
        model.fit(X_train_scaled, y_train[f'target_t{i}'])
        y_pred = model.predict(X_test_scaled)
        predictions[f'target_t{i}'] = y_pred
        metrics[f'target_t{i}'] = evaluate_metrics(y_test[f'target_t{i}'], y_pred)

    print("Evaluation Metrics:")
    for target, metric in metrics.items():
        print(f"{target}: {metric}")

    return metrics, predictions

def plot_results(y_test, predictions):
    """Plot actual vs predicted results."""
    plt.figure(figsize=(15, 10))

    for i, target in enumerate(predictions.keys(), 1):
        plt.subplot(2, 2, i)
        test_index = y_test.index
        pred_index = test_index[:len(predictions[target])]  # Align indices

        plt.plot(test_index, y_test[target], label='Actual', color='blue')
        plt.plot(pred_index, predictions[target], label='Predicted', color='red', linestyle='--')

        plt.title(f'Actual vs Predicted - {target}')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    print("Loading dataset...")
    merged_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/DataPrepared/Load_Weather_Daily_Cleaned_Merged_30_Mins_From_June2016_Till_May_2021_Added_Analysis_Features',
                             index_col='date_time',
                             parse_dates=True)

    print("Preparing data...")
    final_data = prepare_data_with_embeddings(merged_data)

    final_data.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/FinalData_After_Embeddings_Exp1_Code(load, Time2Vec)_21_12_2024.csv')

    print("Training and evaluating models...")
    metrics, predictions = train_and_evaluate_model(final_data)

    print("Plotting results...")
    plot_results(final_data[[f'target_t{i}' for i in range(1, 5)]], predictions)

if __name__ == "__main__":
    main()