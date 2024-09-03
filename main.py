from data_fetcher import fetch_all_data
from data_processor import process_data
from model import train_and_evaluate
import joblib
import mlflow
import mlflow.sklearn
import streamlit as st
import plotly.express as px
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Fetching data...")
    raw_data = fetch_all_data()

    logger.info("Processing data...")
    processed_data = process_data(raw_data)

    logger.info("Training and evaluating model...")
    with mlflow.start_run():
        model, model_name, metrics = train_and_evaluate(processed_data)

        # Log parameters, metrics, and model
        mlflow.log_params({"model_type": model_name})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        logger.info("Model performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")

    logger.info("Saving model...")
    joblib.dump(model, "football_prediction_model.joblib")

    logger.info("Model saved as 'football_prediction_model.joblib'")

    return model, processed_data

def create_dashboard(model, data):
    st.title("Football Match Prediction Dashboard")

    st.header("Model Performance")
    metrics = model.evaluate(data.drop("result", axis=1), data["result"])
    for metric, value in metrics.items():
        st.write(f"{metric.capitalize()}: {value:.4f}")

    st.header("Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame(sorted(zip(importances, data.columns)), columns=['Value','Feature'])
        fig = px.bar(feature_imp, x='Value', y='Feature', orientation='h', title='Feature Importances')
        st.plotly_chart(fig)
    else:
        st.write("Feature importance not available for this model type.")

    st.header("Prediction Tool")
    # Create input fields for key features
    input_data = {}
    for feature in data.columns:
        if feature != "result":
            if data[feature].dtype == 'object':
                input_data[feature] = st.selectbox(feature, data[feature].unique())
            else:
                input_data[feature] = st.number_input(feature, value=float(data[feature].mean()))

    if st.button("Predict"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)
        st.write(f"Predicted outcome: {prediction[0]}")

def main():
    st.sidebar.title("Football Match Prediction")
    page = st.sidebar.selectbox("Choose a page", ["Train Model", "Dashboard"])

    if page == "Train Model":
        st.header("Train Model")
        if st.button("Train New Model"):
            model, data = train_model()
            st.success("Model training completed!")
    elif page == "Dashboard":
        st.header("Model Dashboard")
        try:
            model = joblib.load("football_prediction_model.joblib")
            data = pd.read_csv("processed_data.csv")  # Assuming we save processed data
            create_dashboard(model, data)
        except FileNotFoundError:
            st.error("Model or processed data not found. Please train a model first.")

if __name__ == "__main__":
    main()
