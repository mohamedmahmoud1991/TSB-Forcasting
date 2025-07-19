
# Install necessary libraries
!pip install nltk textblob scikit-learn statsmodels lime econml

# Download necessary NLTK data files
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Download the 'punkt_tab' data package

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

"""GrangerCausality on WordFreq+CountBased

Merge Bnetween SBERT, (WordFreq+Count ) Features reduced after Granger
"""

news_df=pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`14_Selected_Features_After_Granger_25_12_2024.csv')
daily_embeddings=pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`8_News_After_SBertSBert(Combined_Features_For_News)_23_12_2024_After_Aggregating(DailyEmbeddings)_Date_is_the_index.csv')
merged_news_df = pd.merge(daily_embeddings, news_df, on='date', how='inner')
print(merged_news_df.shape)
print(merged_news_df.index.name)
print(merged_news_df.columns.to_list())
print(merged_news_df.head(1))
merged_news_df.to_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`14_(Adding_Time2VEC_SBert)_CombinedColumn_After_Merging_WordFreq_CountBased_SBERT_25_12_2024.csv', index=False)

"""# **FEATURE Combination**
Load Data with News Combined

"""

# Merge Data
news_df=pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/Exp`14_(Adding_Time2VEC_SBert)_CombinedColumn_After_Merging_WordFreq_CountBased_SBERT_25_12_2024.csv')
merged_data = pd.read_csv('/content/drive/MyDrive/FullDataSet/OutPut/FinalData_After_Embeddings_Exp1_Code(load, Time2Vec)_21_12_2024.csv')

print(news_df.shape)
print(news_df.index.name)
print(news_df.columns.to_list())
#print(news_df.head(2))

print(merged_data.shape)
print(merged_data.index.name)
print(merged_data.columns.to_list())
#print(merged_data.head(2))

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