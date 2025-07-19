

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_test, predictions, start_date, end_date):
    mask = (y_test.index >= start_date) & (y_test.index <= end_date)
    y_test_filtered = y_test[mask]
    predictions_filtered = {k: v[mask] for k, v in predictions.items()}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_filtered.index, y_test_filtered['target_t1'], label='Actual Load', color='blue')

    for i, horizon in enumerate(['t1', 't2', 't3', 't4']):
        ax.plot(y_test_filtered.index, predictions_filtered[horizon],
                label=f'Predicted Load (t{i+1})', linestyle='--', color=f'C{i+1}')

    ax.set_title('Actual vs Predicted Load')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
