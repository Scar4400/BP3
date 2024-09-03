import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(df, target_column="result"):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "Neural Network": create_nn_model(X_train.shape[1])
    }

    best_score = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():
        if name == "Neural Network":
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            score = model.evaluate(X_train, y_train, verbose=0)[1]
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5)
            score = np.mean(scores)
            model.fit(X_train, y_train)

        logger.info(f"{name} cross-validation score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    logger.info(f"Best model: {best_model_name}")
    return best_model, best_model_name

def evaluate_model(model, X_test, y_test):
    if isinstance(model, Sequential):
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }

def plot_feature_importance(model, X, output_file="feature_importance.png"):
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(pos, feature_importance[sorted_idx], align="center")
        ax.set_yticks(pos)
        ax.set_yticklabels(X.columns[sorted_idx])
        ax.set_xlabel("Relative Importance")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Feature importance plot saved to {output_file}")
    else:
        logger.warning("Feature importance plot is not available for this model type.")

def train_and_evaluate(df, target_column="result"):
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)
    model, model_name = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train)
    return model, model_name, metrics

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import process_data

    raw_data = fetch_all_data()
    processed_data = process_data(raw_data)
    model, model_name, metrics = train_and_evaluate(processed_data)
    logger.info("Model performance:")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")

