import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load Preprocessor & Models
# --------------------------
@st.cache_resource
def load_pipeline(model_file, preprocessor_file):
    # Load preprocessor
    preprocessor = joblib.load(preprocessor_file)
    # Load trained model
    model = joblib.load(model_file)
    # Build a pipeline: preprocessor + model
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline

# File paths (adjust if needed)
pipelines = {
    "Logistic Regression": load_pipeline("log_model.pkl", "preprocessor.pkl"),
    "Random Forest": load_pipeline("rf_model.pkl", "preprocessor.pkl"),
    "SVM": load_pipeline("svc_model.pkl", "preprocessor.pkl"),
    "KNN": load_pipeline("knn_model.pkl", "preprocessor.pkl"),
    "Naive Bayes": load_pipeline("gnb_model.pkl", "preprocessor.pkl"),
}

# --------------------------
# Streamlit App
# --------------------------
st.title("📊 Customer Churn Prediction")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", list(pipelines.keys()))

# --------------------------
# Collect User Inputs
# --------------------------
st.header("Enter Customer Information")

quantity = st.number_input("Quantity", min_value=0, value=1)
returns = st.number_input("Returns", min_value=0.0, value=0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
purchase_year = st.number_input("Purchase Year", min_value=2000, max_value=2025, value=2023)
purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, value=5)
purchase_day = st.number_input("Purchase Day", min_value=1, max_value=31, value=1)

product_category = st.selectbox("Product Category", ["Books", "Clothing", "Electronics", "Home"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Other"])
age_group = st.selectbox("Age Group", ["0-19", "20-29", "30-39", "40-49", "50-59", "60+"])

total_purchase = st.number_input("Total Purchase Amount (scaled)", value=0.5)
product_price = st.number_input("Product Price (scaled)", value=0.5)

# --------------------------
# Prepare Input DataFrame
# --------------------------
input_dict = {
    "Quantity": quantity,
    "Returns": returns,
    "Gender": 1 if gender == "Male" else 0,
    "purchase_year": purchase_year,
    "purchase_month": purchase_month,
    "purchase_day": purchase_day,
    "Product Category": product_category,
    "Payment Method": payment_method,
    "Age Group": age_group,
    "Total Purchase Amount_scaled": total_purchase,
    "Product Price_scaled": product_price
}

input_df = pd.DataFrame([input_dict])

# --------------------------
# Make Prediction
# --------------------------
if st.button("Predict"):
    pipeline = pipelines[model_choice]
    try:
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN!")
        else:
            st.success("✅ Customer is NOT likely to churn.")

        if proba is not None:
            st.write("Confidence:", f"{max(proba)*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
