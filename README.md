# Customer Churn Prediction System

A full-stack machine learning web application built with **Python**, **Streamlit**, **Pandas**, **NumPy**, **scikit-learn**, and **SQLite** to predict telecom customer churn and visualize model performance.

## Overview

This project uses the Telco Customer Churn dataset to train a **RandomForestClassifier** that predicts whether a customer will churn. The app includes exploratory data analysis (EDA), model training, evaluation metrics, real-time predictions, and SQLite-backed data storage.

## Dataset

- **Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 customer records
- **Target variable:** `Churn` (Yes / No)
- **Problem type:** Binary classification

## Model

| Item | Details |
|------|---------|
| Algorithm | `RandomForestClassifier` |
| Split | 80/20 stratified train/test split |
| Train set | 5,634 records |
| Test set | 1,409 records |
| Features | 32 (after encoding and scaling) |

## Evaluation Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **83.7%** |
| **Accuracy** | **80.0%** |
| Precision | 65.5% |
| Recall | 51.9% |
| F1-score | 57.9% |

Additional evaluation outputs in the app:

- Confusion matrix
- Classification report
- Feature importance plot

## Tech Stack

- Python (OOP)
- Pandas & NumPy
- scikit-learn
- SQLite
- Streamlit
- Matplotlib & Seaborn

## Project Structure

```
customer_churn_prediction_system/
├── app.py                                  # Streamlit app + ML pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
├── requirements.txt                        # Python dependencies
├── churn.db                                # SQLite database (generated at runtime)
└── README.md
```

## Setup & Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Open the app in your browser and navigate to the **Model Training & Performance** tab to train the model and view metrics.

## Features

- **EDA:** Churn distribution, tenure analysis, contract-based churn rates, correlation heatmap
- **Model Training:** Random Forest with preprocessing pipeline (cleaning, encoding, scaling)
- **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance
- **Prediction:** Interactive form for real-time churn scoring
- **SQL Integration:** Stores raw data, cleaned data, and predictions in SQLite

## Author

Govind Sharma
