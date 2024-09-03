import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

def preprocess_data(data):
    dfs = {}
    for source, content in data.items():
        if isinstance(content, dict) and 'data' in content:
            dfs[source] = pd.DataFrame(content['data'])
        elif isinstance(content, dict):
            dfs[source] = pd.DataFrame.from_dict(content, orient='index').reset_index()
            dfs[source].columns = ['team', f'{source}_score']

    df = pd.DataFrame()
    for source, source_df in dfs.items():
        if 'team' in source_df.columns:
            df = pd.merge(df, source_df, on='team', how='outer') if not df.empty else source_df

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df

def engineer_features(df):
    # Create time-based features if timestamp is available
    if 'timestamp' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Calculate rolling averages
    for col in df.columns:
        if col.endswith('_score'):
            df[f"{col}_rolling_avg"] = df.groupby("team")[col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

    # Create interaction features
    score_columns = [col for col in df.columns if col.endswith('_score')]
    for i in range(len(score_columns)):
        for j in range(i+1, len(score_columns)):
            df[f"{score_columns[i]}_{score_columns[j]}_interaction"] = df[score_columns[i]] * df[score_columns[j]]

    # Create team embeddings
    team_descriptions = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
    tokenized_descriptions = [word_tokenize(desc.lower()) for desc in team_descriptions]
    model = Word2Vec(sentences=tokenized_descriptions, vector_size=10, window=5, min_count=1, workers=4)

    team_embeddings = {team: model.wv[team] for team in df["team"].unique() if team in model.wv}
    embedding_df = pd.DataFrame.from_dict(team_embeddings, orient='index')
    embedding_df.columns = [f'team_embedding_{i}' for i in range(10)]

    df = pd.merge(df, embedding_df, left_on="team", right_index=True, how="left")

    return df

def scale_features(df):
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def process_data(raw_data):
    logger.info("Preprocessing data...")
    df = preprocess_data(raw_data)
    logger.info("Engineering features...")
    df = engineer_features(df)
    logger.info("Scaling features...")
    df = scale_features(df)
    return df

if __name__ == "__main__":
    from data_fetcher import fetch_all_data

    raw_data = fetch_all_data()
    processed_data = process_data(raw_data)
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Processed data columns: {processed_data.columns}")

