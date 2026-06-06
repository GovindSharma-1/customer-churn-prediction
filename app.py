import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def inject_custom_css() -> None:
    """Global CSS for a modern SaaS-like dark dashboard."""
    st.markdown(
        """
        <style>
            :root {
                --bg-primary: #070d1b;
                --bg-card: #0f1b2d;
                --bg-card-2: #11253f;
                --text-primary: #e8f2ff;
                --text-muted: #96a7c1;
                --accent: #11d4c2;
                --accent-2: #36e6d4;
                --border: #1f3555;
                --shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            }

            .stApp {
                background: radial-gradient(circle at 10% 10%, #11223c 0%, var(--bg-primary) 50%, #040811 100%);
                color: var(--text-primary);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #081121 0%, #0a1629 100%);
                border-right: 1px solid var(--border);
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(17, 37, 63, 0.95), rgba(7, 13, 27, 0.95));
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                margin: 0;
                color: var(--text-primary);
                letter-spacing: 0.2px;
            }

            .hero-sub {
                margin-top: 0.35rem;
                color: var(--text-muted);
                font-size: 1rem;
            }

            .sidebar-card {
                background: linear-gradient(160deg, #0d1d34, #0a1322);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.8rem;
            }

            .tech-list li {
                margin: 0.25rem 0;
                color: #d7e7ff;
            }

            div[data-baseweb="tab-list"] {
                gap: 0.5rem;
                background: rgba(10, 19, 34, 0.55);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 0.4rem;
            }

            button[data-baseweb="tab"] {
                border-radius: 10px !important;
                padding: 0.5rem 1rem !important;
                color: #b8d2f3 !important;
                transition: all 0.2s ease-in-out;
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(90deg, #0f7e87, #11d4c2) !important;
                color: #061322 !important;
                font-weight: 700;
            }

            .stButton > button, .stFormSubmitButton > button {
                border: 0 !important;
                border-radius: 12px !important;
                background: linear-gradient(90deg, #0e8ea3 0%, #11d4c2 100%) !important;
                color: #05111f !important;
                font-weight: 700 !important;
                box-shadow: 0 8px 20px rgba(17, 212, 194, 0.25);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .stButton > button:hover, .stFormSubmitButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 24px rgba(17, 212, 194, 0.35);
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(165deg, #0d1b30, #0d2037);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.4rem 0.7rem;
                box-shadow: var(--shadow);
            }

            div[data-testid="stDataFrame"], div[data-testid="stCodeBlock"] {
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class ModelResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_size: int
    test_size: int
    confusion_matrix_data: np.ndarray
    report: str
    feature_importance: pd.Series


class ChurnPredictor:
    """
    OOP core class for end-to-end churn workflow:
    loading -> preprocessing -> training -> evaluation -> prediction -> SQL persistence.
    """

    def __init__(self, csv_path: str, db_path: str = "churn.db") -> None:
        self.csv_path = csv_path
        self.db_path = db_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[pd.Index] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def load_data(self) -> pd.DataFrame:
        """Load CSV data using Pandas and save raw copy to SQLite."""
        try:
            df = pd.read_csv(self.csv_path)
            self.raw_df = df.copy()
            self.save_to_sqlite("raw_data", self.raw_df)
            return df
        except Exception as exc:
            raise RuntimeError(f"Error loading CSV file: {exc}") from exc

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Heavy Pandas + NumPy preprocessing:
        - clean TotalCharges
        - drop duplicates / handle missing values
        - feature engineering
        - one-hot encoding
        - scaling
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df.copy()

        # Pandas string cleanup + numeric conversion for inconsistent column type.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

        # Pandas data hygiene.
        df = df.drop_duplicates()
        df = df.dropna(subset=["Churn"])

        # fillna for numeric and categorical columns.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if "customerID" in categorical_cols:
            categorical_cols.remove("customerID")

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

        # NumPy feature engineering example (binary flags from thresholds).
        df["HighMonthlyCharges"] = np.where(df["MonthlyCharges"] > 80, 1, 0)
        df["LongTenure"] = np.where(df["tenure"] >= 24, 1, 0)

        # Convert target to binary.
        y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

        # Drop non-feature columns.
        X = df.drop(columns=["Churn", "customerID"], errors="ignore")

        # Pandas one-hot encoding for categorical features.
        X = pd.get_dummies(X, drop_first=True)
        self.feature_columns = X.columns

        # Scaling numeric features.
        self.scaler = StandardScaler()
        scaled_array = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)

        self.cleaned_df = df.copy()
        self.save_to_sqlite("cleaned_data", self.cleaned_df)

        return X_scaled, y

    def train_model(self) -> None:
        """Train RandomForestClassifier on prepared data."""
        X, y = self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self) -> ModelResults:
        """Evaluate model with classification metrics, confusion matrix, and feature importance."""
        if self.model is None or self.X_test is None or self.y_test is None:
            self.train_model()

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )

        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.X_train.columns,
        ).sort_values(ascending=False)

        return ModelResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            train_size=len(self.X_train),
            test_size=len(self.X_test),
            confusion_matrix_data=cm,
            report=report,
            feature_importance=feature_importance,
        )

    def predict(self, input_data: Dict) -> Tuple[int, float]:
        """
        Predict churn label and probability for one customer row.
        Returns: (prediction_label, churn_probability)
        """
        if self.model is None or self.scaler is None or self.feature_columns is None:
            self.train_model()

        input_df = pd.DataFrame([input_data])
        processed_input = pd.get_dummies(input_df, drop_first=True)
        processed_input = processed_input.reindex(columns=self.feature_columns, fill_value=0)
        processed_input_scaled = self.scaler.transform(processed_input)

        pred = int(self.model.predict(processed_input_scaled)[0])
        prob = float(self.model.predict_proba(processed_input_scaled)[0][1])

        # SQL integration for storing predictions.
        pred_record = pd.DataFrame(
            [
                {
                    **input_data,
                    "prediction": pred,
                    "churn_probability": prob,
                }
            ]
        )
        self.save_to_sqlite("predictions", pred_record, if_exists="append")
        return pred, prob

    def save_to_sqlite(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
        """Save DataFrame to SQLite table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        except Exception as exc:
            raise RuntimeError(f"Error saving table '{table_name}' to SQLite: {exc}") from exc

    def load_from_sqlite(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from SQLite table with optional row limit."""
        try:
            query = f"SELECT * FROM {table_name}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as exc:
            raise RuntimeError(f"Error loading table '{table_name}' from SQLite: {exc}") from exc


def render_sidebar() -> None:
    st.sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<h2 style='text-align:center; margin:0;'>📊 Churn Console</h2>"
        "<p style='text-align:center; color:#9eb0cb; margin:0.25rem 0 0;'>ML Intelligence Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<h4 style='margin:0; text-align:center;'>Technologies Used</h4>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<ul class='tech-list'>"
        "<li>Python OOP</li>"
        "<li>NumPy</li>"
        "<li>Pandas</li>"
        "<li>SQLite (SQL)</li>"
        "<li>Scikit-learn ML</li>"
        "<li>Streamlit</li>"
        "</ul>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


def render_eda_tab(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    st.caption("📈 Understand customer behavior patterns before modeling.")
    st.dataframe(df.head(10), use_container_width=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", palette="Set2", ax=ax1)
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(df["tenure"], kde=True, bins=30, ax=ax2, color="teal")
    ax2.set_title("Tenure Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="pastel", ax=ax3)
    ax3.set_title("Monthly Charges vs Churn")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    churn_by_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    churn_by_contract.plot(kind="bar", stacked=True, ax=ax4, colormap="viridis")
    ax4.set_ylabel("Percentage")
    ax4.set_title("Churn % by Contract Type")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    corr_data = df[numeric_cols].corr(numeric_only=True)
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
    ax5.set_title("Numeric Feature Correlation Heatmap")
    st.pyplot(fig5)


def render_model_tab(predictor: ChurnPredictor) -> None:
    st.subheader("Model Training & Performance")
    st.caption("🧠 Train, evaluate, and interpret the churn model.")
    if st.button("Train / Retrain Model", use_container_width=True):
        with st.spinner("Training model..."):
            predictor.train_model()
            st.success("Model training completed successfully.")

    if predictor.model is not None:
        results = predictor.evaluate_model()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{results.accuracy:.2%}")
        col2.metric("Precision", f"{results.precision:.2%}")
        col3.metric("Recall", f"{results.recall:.2%}")
        col4.metric("F1 Score", f"{results.f1:.2%}")
        col5.metric("ROC-AUC", f"{results.roc_auc:.2%}")

        st.caption(
            f"Train set: {results.train_size:,} rows | Test set: {results.test_size:,} rows "
            f"(80/20 stratified split)"
        )

        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            results.confusion_matrix_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.markdown("#### Classification Report")
        st.code(results.report)

        st.markdown("#### Top 15 Feature Importances")
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        results.feature_importance.head(15).sort_values().plot(kind="barh", ax=ax_imp, color="slateblue")
        ax_imp.set_xlabel("Importance Score")
        st.pyplot(fig_imp)
    else:
        st.info("Click the train button to build the model and view performance.")


def render_prediction_tab(predictor: ChurnPredictor) -> None:
    st.subheader("Make New Prediction")
    st.caption("✨ Fill customer profile details and generate real-time churn risk.")
    st.markdown(
        "<div class='hero-card' style='padding:0.9rem 1.1rem;'>"
        "<b>Prediction Studio</b><br>"
        "<span style='color:#9eb0cb;'>Interactive form with engineered features and live probability output.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col1.selectbox("Senior Citizen", [0, 1])
        partner = col1.selectbox("Partner", ["No", "Yes"])
        dependents = col1.selectbox("Dependents", ["No", "Yes"])

        tenure = col2.slider("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = col2.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = col2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = col2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        online_security = col3.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = col3.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = col3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = col3.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = col3.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = col3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=850.0, step=10.0)

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        input_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "HighMonthlyCharges": 1 if monthly_charges > 80 else 0,
            "LongTenure": 1 if tenure >= 24 else 0,
        }

        try:
            pred, prob = predictor.predict(input_payload)
            if pred == 1:
                st.error(f"⚠️ Prediction: Customer is likely to churn ({prob:.2%} probability).")
            else:
                st.success(f"✅ Prediction: Customer is likely to stay ({1 - prob:.2%} confidence).")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption("💾 Prediction saved into SQLite table: predictions")
        except Exception as exc:
            st.exception(exc)


def render_sql_tab(predictor: ChurnPredictor) -> None:
    st.subheader("SQLite Integration")
    st.caption("🗄️ Query and monitor raw, cleaned, and prediction data.")
    st.write("Database file: `churn.db`")

    st.markdown("#### Sample SQL Queries")
    st.code(
        "SELECT COUNT(*) AS total_customers FROM raw_data;\n"
        "SELECT Churn, COUNT(*) AS cnt FROM cleaned_data GROUP BY Churn;\n"
        "SELECT prediction, AVG(churn_probability) AS avg_prob FROM predictions GROUP BY prediction;\n"
        "SELECT * FROM predictions ORDER BY rowid DESC LIMIT 5;"
    )

    st.markdown("#### Preview Table Data")
    table_name = st.selectbox("Select table", ["raw_data", "cleaned_data", "predictions"])
    limit = st.slider("Rows to display", min_value=5, max_value=50, value=10)

    try:
        preview_df = predictor.load_from_sqlite(table_name=table_name, limit=limit)
        st.dataframe(preview_df, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not read table '{table_name}': {exc}")


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction System", layout="wide")
    inject_custom_css()
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">📉 Customer Churn Prediction System</h1>
            <div class="hero-sub">Deep analytics, ML performance tracking, and real-time churn scoring in one dashboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()
    predictor = ChurnPredictor(csv_path="WA_Fn-UseC_-Telco-Customer-Churn.csv", db_path="churn.db")

    try:
        data = predictor.load_data()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Model Training & Performance", "Make New Prediction", "SQL Data"])

    with tab1:
        render_eda_tab(data)

    with tab2:
        render_model_tab(predictor)

    with tab3:
        render_prediction_tab(predictor)

    with tab4:
        render_sql_tab(predictor)


if __name__ == "__main__":
    main()

# requirements.txt content:
# streamlit
# pandas
# numpy
# scikit-learn
# matplotlib
# seaborn
