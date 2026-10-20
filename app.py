
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Fraud Detection in Banking")
st.write("Machine learning-based credit card transaction analysis")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_model(data):
    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both fraud and legitimate transactions.")

    X = X.apply(pd.to_numeric, errors="raise")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, X.columns.tolist(), X_test, y_test, score

uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

if uploaded_file:
    try:
        data = load_data(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        if "Class" not in data.columns:
            st.error("The dataset needs a 'Class' target column.")
            st.stop()

        fraud_count = int((data["Class"] == 1).sum())
        normal_count = int((data["Class"] == 0).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{len(data):,}")
        col2.metric("Legitimate Transactions", f"{normal_count:,}")
        col3.metric("Fraudulent Transactions", f"{fraud_count:,}")

        st.subheader("Transaction Distribution")
        st.bar_chart(
            pd.DataFrame({
                "Transaction count": [
                    normal_count, fraud_count
                ]
            }, index=["Legitimate", "Fraudulent"])
        )

        if st.button("Train Fraud Detection Model"):
            with st.spinner("Training model..."):
                model, features, X_test, y_test, score = train_model(data)

            st.session_state["fraud_model"] = model
            st.session_state["fraud_features"] = features
            st.session_state["fraud_score"] = score
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

        if "fraud_model" in st.session_state:
            st.success("Model training completed.")
            st.metric(
                "Holdout Accuracy",
                f"{st.session_state['fraud_score']:.2%}"
            )

            st.caption(
                "Accuracy alone may be misleading for imbalanced fraud data. "
                "Review precision, recall and the confusion matrix before "
                "evaluating model performance."
            )

            if st.button("Show Evaluation Report"):
                from sklearn.metrics import (
                    classification_report,
                    confusion_matrix
                )

                model = st.session_state["fraud_model"]
                predictions = model.predict(st.session_state["X_test"])

                st.text(classification_report(
                    st.session_state["y_test"],
                    predictions,
                    zero_division=0
                ))

                st.write("Confusion matrix")
                st.dataframe(pd.DataFrame(
                    confusion_matrix(
                        st.session_state["y_test"], predictions
                    ),
                    index=["Actual legitimate", "Actual fraud"],
                    columns=["Predicted legitimate", "Predicted fraud"]
                ))

            st.subheader("Test a Transaction")
            st.info(
                "Enter values for the same features used by the dataset. "
                "This demo is for educational purposes only."
            )

            with st.form("transaction_form"):
                values = {}
                columns = st.columns(3)

                for i, feature in enumerate(
                    st.session_state["fraud_features"]
                ):
                    with columns[i % 3]:
                        values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.6f"
                        )

                submitted = st.form_submit_button(
                    "Check Transaction"
                )

            if submitted:
                sample = pd.DataFrame(
                    [values],
                    columns=st.session_state["fraud_features"]
                )
                prediction = st.session_state[
                    "fraud_model"
                ].predict(sample)[0]

                if prediction == 1:
                    st.error(
                        "⚠️ Model flagged this transaction as potentially fraudulent."
                    )
                else:
                    st.success(
                        "Model classified this transaction as likely legitimate."
                    )

                st.caption(
                    "This prediction is not a definitive fraud determination."
                )

    except Exception as error:
        st.error(f"Could not process the dataset: {error}")

else:
    st.info(
        "Upload a labeled credit card dataset containing a 'Class' column. "
        "Class 1 should indicate fraud and Class 0 legitimate transactions."
    )