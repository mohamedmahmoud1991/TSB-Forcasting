

import pandas as pd
import numpy as np
from forecaster import LoadForecaster
from visuals import plot_predictions  # optional

if __name__ == "__main__":
    # === CONFIGURABLE PARAMETERS ===
    data_path = 'data/your_dataset.csv'
    date_column = 'date_time'
    target_column = 'actual_load_MW'
    train_split_date = '2020-04-30'
    meta_split_date = '2019-12-31'
    forecast_plot_start = '2020-05-01'
    forecast_plot_end = '2020-05-07'
    top_n_features = 10

    # === PIPELINE ===
    forecaster = LoadForecaster()

    # Load and prepare data
    final_data = pd.read_csv(data_path)
    processed_data = forecaster.prepare_data(final_data)

    # Time-based split
    train_data, test_data = forecaster.create_train_test_split(processed_data)

    # Meta-learner split
    train_base = train_data[train_data.index <= meta_split_date]
    train_meta = train_data[train_data.index > meta_split_date]

    # Scale
    X_base_scaled, X_meta_scaled = forecaster.scale_features(train_base, train_meta)
    _, X_test_scaled = forecaster.scale_features(train_data, test_data)

    # Targets
    y_base = train_base[[f'target_t{i}' for i in range(1, 5)]]
    y_meta = train_meta[[f'target_t{i}' for i in range(1, 5)]]
    y_test = test_data[[f'target_t{i}' for i in range(1, 5)]]

    # Train models and meta-learner
    forecaster.train_models(X_base_scaled, y_base)
    forecaster.train_meta_learner(X_meta_scaled, y_meta)

    # Predict and evaluate
    predictions, results = forecaster.predict_and_evaluate(X_test_scaled, y_test)
    top_features = forecaster.get_top_feature_importance(top_n=top_n_features)

    # Summary
    print("\nSummary of Predictions and Metrics:")
    for horizon, metrics in results.items():
        print(f"\nHorizon: {horizon}")
        print(f"  Mean Prediction: {np.mean(predictions[horizon]):.4f}")
        print(f"  Std of Predictions: {np.std(predictions[horizon]):.4f}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Plot
    plot_predictions(y_test, predictions, forecast_plot_start, forecast_plot_end)

    # Avg metrics
    avg = forecaster.calculate_avg_metrics(results)
    print("\nAverage Metrics Across All Horizons:")
    for metric, value in avg.items():
        print(f"{metric}: {value:.4f}")
