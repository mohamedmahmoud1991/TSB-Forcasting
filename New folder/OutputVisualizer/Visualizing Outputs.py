

# Install Graphviz if not already installed
!apt-get install graphviz -y
!pip install graphviz

import graphviz

# Define the directed graph
flowchart = graphviz.Digraph(format="png", engine="dot")

# Data Sources
flowchart.node("A", "ENTSO-E (Energy Data)", shape="box", style="filled", fillcolor="lightblue")
flowchart.node("B", "BBC (News Data)", shape="box", style="filled", fillcolor="lightcoral")

# Preprocessing
flowchart.node("C", "Time-Series Preprocessing", shape="box", style="filled", fillcolor="lightblue")
flowchart.node("D", "Text Preprocessing", shape="box", style="filled", fillcolor="lightcoral")

# Feature Engineering
flowchart.node("E", "Feature Engineering", shape="box", style="filled", fillcolor="gold")
flowchart.node("F", "Time2Vec Embeddings", shape="ellipse", style="filled", fillcolor="gold")
flowchart.node("G", "SBERT Embeddings", shape="ellipse", style="filled", fillcolor="gold")

# Model Building
flowchart.node("H", "Model Building", shape="box", style="filled", fillcolor="lightgreen")
flowchart.node("I", "Extra Trees Regressor (ETR)", shape="ellipse", style="filled", fillcolor="lightgreen")
flowchart.node("J", "XGBoost", shape="ellipse", style="filled", fillcolor="lightgreen")
flowchart.node("K", "Meta Learner", shape="ellipse", style="filled", fillcolor="lightgreen")
flowchart.node("L", "Final Prediction", shape="box", style="filled", fillcolor="cyan")

# Connect Nodes (Data Flow)
flowchart.edge("A", "C", label="Raw Energy Data")
flowchart.edge("B", "D", label="Raw News Data")
flowchart.edge("C", "E", label="Processed Time Data")
flowchart.edge("D", "E", label="Processed Text Data")
flowchart.edge("E", "F", label="Extracted Time-Series Features")
flowchart.edge("E", "G", label="Extracted Text Features")
flowchart.edge("F", "H", label="Input to Model")
flowchart.edge("G", "H", label="Input to Model")
flowchart.edge("H", "I", label="Train ETR")
flowchart.edge("H", "J", label="Train XGBoost")
flowchart.edge("I", "K", label="ETR Predictions")
flowchart.edge("J", "K", label="XGBoost Predictions")
flowchart.edge("K", "L", label="Final Forecast")

# Render and display
flowchart.render("data_flowchart", format="png", cleanup=False)
from IPython.display import Image
Image(filename="data_flowchart.png")

import matplotlib.pyplot as plt

# Data for the experiment
horizons = ['2h', '2d', '7d', '15d', '30d']
rmse = [1678.55, 2331.00, 2570.68, 2947.62, 3800.81]
mae = [1238.42, 1768.74, 1962.18, 2228.83, 2907.36]
smape = [3.79, 5.46, 6.00, 6.72, 8.73]

# Plot RMSE and MAE in a separate figure with higher resolution and larger font sizes
plt.figure(figsize=(10, 6), dpi=200)  # Increased DPI for better resolution
plt.plot(horizons, rmse, label='RMSE', marker='o', color='blue')
plt.plot(horizons, mae, label='MAE', marker='o', color='orange')
plt.title('RMSE and MAE Across Different Forecast Horizons', fontsize=16)  # Larger title font size
plt.xlabel('Forecast Horizon', fontsize=14)  # Larger x-axis label font size
plt.ylabel('Error Value', fontsize=14)  # Larger y-axis label font size
plt.legend(fontsize=12)  # Larger legend font size
plt.xticks(fontsize=12)  # Larger x-axis tick font size
plt.yticks(fontsize=12)  # Larger y-axis tick font size
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot SMAPE in a separate figure with higher resolution and larger font sizes
plt.figure(figsize=(10, 6), dpi=200)  # Increased DPI for better resolution
plt.plot(horizons, smape, label='SMAPE', marker='o', color='green')
plt.title('SMAPE Across Different Forecast Horizons', fontsize=16)  # Larger title font size
plt.xlabel('Forecast Horizon', fontsize=14)  # Larger x-axis label font size
plt.ylabel('SMAPE (%)', fontsize=14)  # Larger y-axis label font size
plt.legend(fontsize=12)  # Larger legend font size
plt.xticks(fontsize=12)  # Larger x-axis tick font size
plt.yticks(fontsize=12)  # Larger y-axis tick font size
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Define models and updated metric values in MW for MAE and RMSE, and percentage for SMAPE
models = ["ENTSO-E", "ETR", "M6", "TSB-Forecast"]
mae_scores = [2423.56, 1844.55, 1793.21, 1238.42]  # MAE in MW
rmse_scores = [2484.74, 1832.49, 1841.04, 1678.55]  # RMSE in MW
smape_scores = [9.27, 6.99, 6.78, 3.79]  # SMAPE in %

# Define bar colors for each model
colors = ['gray', 'blue', 'orange', 'green']

# Plot MAE Comparison (in MW)
plt.figure(figsize=(6, 4))
plt.bar(models, mae_scores, color=colors)
plt.title("MAE Comparison")
plt.ylabel("Mean Absolute Error (MW)")
plt.ylim(0, max(mae_scores) + 500)  # Adjust y-axis dynamically
plt.show()

# Plot RMSE Comparison (in MW)
plt.figure(figsize=(6, 4))
plt.bar(models, rmse_scores, color=colors)
plt.title("RMSE Comparison")
plt.ylabel("Root Mean Squared Error (MW)")
plt.ylim(0, max(rmse_scores) + 500)  # Adjust y-axis dynamically
plt.show()

# Plot SMAPE Comparison (in %)
plt.figure(figsize=(6, 4))
plt.bar(models, smape_scores, color=colors)
plt.title("SMAPE Comparison")
plt.ylabel("Symmetric Mean Absolute Percentage Error (%)")
plt.ylim(0, max(smape_scores) + 2)  # Adjust y-axis dynamically
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Metrics summary for each fold (mean and std values)
folds = [1, 2, 3, 5, 7, 10]
mae_means = [1245.00, 952.46, 962.08, 1110.31, 1106.53, 1014.73]
mae_stds  = [0.00, 172.30, 198.40, 369.69, 454.54, 267.31]

rmse_means = [2064.23, 2186.12, 2055.36, 2162.01, 2100.57, 1982.11]
rmse_stds  = [0.00, 235.99, 600.64, 411.65, 648.17, 535.77]

smape_means = [3.86, 3.19, 3.13, 3.42, 3.42, 3.18]
smape_stds  = [0.00, 0.69, 0.71, 0.82, 1.14, 0.71]

# Setup bar plot
x = np.arange(len(folds))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, mae_means, width=width, yerr=mae_stds, capsize=5, label='MAE')
plt.bar(x, rmse_means, width=width, yerr=rmse_stds, capsize=5, label='RMSE')
plt.bar(x + width, smape_means, width=width, yerr=smape_stds, capsize=5, label='SMAPE')

plt.xticks(x, [f"{f}-Fold" for f in folds])
plt.xlabel('Fold Setting')
plt.ylabel('Metric Value')
plt.title('Comparison of Evaluation Metrics Across Folds (with STD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the metrics with and without Time2Vec
metrics_mw = ["RMSE (MW)", "MAE (MW)"]
metrics_percent = ["SMAPE (%)"]
with_time2vec_mw = [1767.7946, 1116.762]
without_time2vec_mw = [2507.1248, 1763.8106]
with_time2vec_smape = [3.7585]
without_time2vec_smape = [5.9064]

# Define bar width
bar_width = 0.4
x_mw = np.arange(len(metrics_mw))
x_percent = np.arange(len(metrics_percent))

# Define bar colors
colors = ["green", "red"]

# Create bar chart for RMSE and MAE (MW)
plt.figure(figsize=(6, 4))
plt.bar(x_mw - bar_width/2, with_time2vec_mw, bar_width, label="With Time2Vec", color=colors[0])
plt.bar(x_mw + bar_width/2, without_time2vec_mw, bar_width, label="Without Time2Vec", color=colors[1])
plt.xticks(x_mw, metrics_mw)
plt.ylabel("Error in MW")
plt.title("Impact of Time2Vec on RMSE & MAE")
plt.legend()
plt.show()

# Create bar chart for SMAPE (%)
plt.figure(figsize=(6, 4))
plt.bar(x_percent - bar_width/2, with_time2vec_smape, bar_width, label="With Time2Vec", color=colors[0])
plt.bar(x_percent + bar_width/2, without_time2vec_smape, bar_width, label="Without Time2Vec", color=colors[1])
plt.xticks(x_percent, metrics_percent)
plt.ylabel("Error in %")
plt.title("Impact of Time2Vec on SMAPE")
plt.legend()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the metrics with and without Time2Vec
metrics_mw = ["RMSE (MW)", "MAE (MW)"]
metrics_percent = ["SMAPE (%)"]
with_time2vec_mw = [1767.7946, 1116.762]
without_time2vec_mw = [2507.1248, 1763.8106]
with_time2vec_smape = [3.7585]
without_time2vec_smape = [5.9064]

# Define x positions for grouped plots
x_mw = np.arange(len(metrics_mw))
x_percent = np.arange(len(metrics_percent))

# Define colors
colors = ["green", "red"]

# 1️⃣ Line Plot (Trend Comparison)
plt.figure(figsize=(6, 4))
plt.plot(metrics_mw, with_time2vec_mw, marker="o", linestyle="-", color=colors[0], label="With Time2Vec")
plt.plot(metrics_mw, without_time2vec_mw, marker="s", linestyle="--", color=colors[1], label="Without Time2Vec")
plt.ylabel("Error in MW")
plt.title("Impact of Time2Vec on RMSE & MAE (Line Chart)")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(metrics_percent, with_time2vec_smape, marker="o", linestyle="-", color=colors[0], label="With Time2Vec")
plt.plot(metrics_percent, without_time2vec_smape, marker="s", linestyle="--", color=colors[1], label="Without Time2Vec")
plt.ylabel("Error in %")
plt.title("Impact of Time2Vec on SMAPE (Line Chart)")
plt.legend()
plt.show()

# 2️⃣ Grouped Box Plot (Distribution Analysis)
plt.figure(figsize=(6, 4))
sns.boxplot(data=[with_time2vec_mw, without_time2vec_mw], palette=colors)
plt.xticks([0, 1], ["With Time2Vec", "Without Time2Vec"])
plt.ylabel("Error in MW")
plt.title("Distribution of RMSE & MAE with/without Time2Vec (Box Plot)")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=[with_time2vec_smape, without_time2vec_smape], palette=colors)
plt.xticks([0, 1], ["With Time2Vec", "Without Time2Vec"])
plt.ylabel("Error in %")
plt.title("Distribution of SMAPE with/without Time2Vec (Box Plot)")
plt.show()

# 3️⃣ Heatmap (Relative Error Reduction)
data = np.array([[with_time2vec_mw[0], without_time2vec_mw[0]],
                 [with_time2vec_mw[1], without_time2vec_mw[1]],
                 [with_time2vec_smape[0], without_time2vec_smape[0]]])

plt.figure(figsize=(6, 4))
sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=["With Time2Vec", "Without Time2Vec"],
            yticklabels=["RMSE (MW)", "MAE (MW)", "SMAPE (%)"], linewidths=0.5)
plt.title("Error Reduction with Time2Vec (Heatmap)")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the metrics
metrics_mw = ["RMSE (MW)", "MAE (MW)"]
metrics_percent = ["SMAPE (%)"]

# SBERT performance scores
with_sbert_mw = [1660, 1020]
without_sbert_mw = [2507, 1763]
with_sbert_smape = [3.3898]
without_sbert_smape = [5.9]

# Define bar width and positions
bar_width = 0.4
x_mw = np.arange(len(metrics_mw))
x_percent = np.arange(len(metrics_percent))

# Define bar colors
colors = ["green", "red"]

# Bar chart for RMSE and MAE (MW)
plt.figure(figsize=(6, 4))
plt.bar(x_mw - bar_width/2, with_sbert_mw, bar_width, label="With SBERT", color=colors[0])
plt.bar(x_mw + bar_width/2, without_sbert_mw, bar_width, label="Without SBERT", color=colors[1])
plt.xticks(x_mw, metrics_mw)
plt.ylabel("Error in MW")
plt.title("Impact of SBERT on RMSE & MAE")
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for SMAPE (%)
plt.figure(figsize=(6, 4))
plt.bar(x_percent - bar_width/2, with_sbert_smape, bar_width, label="With SBERT", color=colors[0])
plt.bar(x_percent + bar_width/2, without_sbert_smape, bar_width, label="Without SBERT", color=colors[1])
plt.xticks(x_percent, metrics_percent)
plt.ylabel("Error in %")
plt.title("Impact of SBERT on SMAPE")
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the metrics with and without SBERT + Time2Vec
metrics_mw = ["RMSE (MW)", "MAE (MW)"]
metrics_percent = ["SMAPE (%)"]
with_sbert_time2vec_mw = [1401.0689, 807.4228]
without_sbert_time2vec_mw = [1844.55, 1832.49]
with_sbert_time2vec_smape = [2.7511]
without_sbert_time2vec_smape = [6.99]

# Define x positions for grouped plots
x_mw = np.arange(len(metrics_mw))
x_percent = np.arange(len(metrics_percent))

# Define colors
colors = ["green", "red"]

# 1️⃣ Line Plot for RMSE and MAE (MW)
plt.figure(figsize=(6, 4))
plt.plot(metrics_mw, with_sbert_time2vec_mw, marker="o", linestyle="-", color=colors[0], label="With SBERT + Time2Vec")
plt.plot(metrics_mw, without_sbert_time2vec_mw, marker="s", linestyle="--", color=colors[1], label="Without SBERT + Time2Vec")
plt.ylabel("Error in MW")
plt.title("Impact of Combining SBERT & Time2Vec on RMSE & MAE")
plt.legend()
plt.show()

# 2️⃣ Bar Chart for SMAPE (%)
plt.figure(figsize=(6, 4))
plt.bar(x_percent - 0.2, with_sbert_time2vec_smape, 0.4, label="With SBERT + Time2Vec", color=colors[0])
plt.bar(x_percent + 0.2, without_sbert_time2vec_smape, 0.4, label="Without SBERT + Time2Vec", color=colors[1])
plt.xticks(x_percent, metrics_percent)
plt.ylabel("Error in %")
plt.title("Impact of Combining SBERT & Time2Vec on SMAPE")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define models for comparison
models = ["CNN-GRU", "TSB-Forecast"]

# Performance Metrics
smape_scores = [22.55, 3.79]  # SMAPE in %
rmse_scores = [2388.24, 1678.55]  # RMSE in MW
mae_scores = [1897.62, 1238.44]  # MAE in MW

# Colors for the charts
colors = ['red', 'green']

# 1️⃣ Bar Chart for SMAPE (%)
plt.figure(figsize=(6, 4))
plt.bar(models, smape_scores, color=colors)
plt.title("SMAPE Comparison: TSB-Forecast vs. CNN-GRU")
plt.ylabel("SMAPE (%)")
plt.ylim(0, max(smape_scores) + 5)
plt.show()

# 2️⃣ Bar Chart for RMSE (MW)
plt.figure(figsize=(6, 4))
plt.bar(models, rmse_scores, color=colors)
plt.title("RMSE Comparison: TSB-Forecast vs. CNN-GRU")
plt.ylabel("RMSE (MW)")
plt.ylim(0, max(rmse_scores) + 300)
plt.show()

# 3️⃣ Bar Chart for MAE (MW)
plt.figure(figsize=(6, 4))
plt.bar(models, mae_scores, color=colors)
plt.title("MAE Comparison: TSB-Forecast vs. CNN-GRU")
plt.ylabel("MAE (MW)")
plt.ylim(0, max(mae_scores) + 300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Metrics summary for each fold (mean and std values)
folds = [2, 3, 5, 7, 10]
mae_means = [ 952.46, 962.08, 1110.31, 1106.53, 1014.73]
mae_stds  = [172.30, 198.40, 369.69, 454.54, 267.31]

rmse_means = [ 2186.12, 2055.36, 2162.01, 2100.57, 1982.11]
rmse_stds  = [235.99, 600.64, 411.65, 648.17, 535.77]

smape_means = [3.19, 3.13, 3.42, 3.42, 3.18]
smape_stds  = [ 0.69, 0.71, 0.82, 1.14, 0.71]

# Setup bar plot
x = np.arange(len(folds))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, mae_means, width=width, yerr=mae_stds, capsize=5, label='MAE')
plt.bar(x, rmse_means, width=width, yerr=rmse_stds, capsize=5, label='RMSE')
plt.bar(x + width, smape_means, width=width, yerr=smape_stds, capsize=5, label='SMAPE')

plt.xticks(x, [f"{f}-Fold" for f in folds])
plt.xlabel('Fold Setting')
plt.ylabel('Metric Value')
plt.title('Comparison of Evaluation Metrics Across Folds (with STD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()