import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------
# Load Saved Models & Preprocessor
# --------------------------
@st.cache_resource
def load_models_and_preprocessor():
    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("gnb_model.pkl"),
    }
    preprocessor = joblib.load("preprocessor.pkl")
    
    # Load expected categorical levels to fix missing column issues
    ohe_features = joblib.load("ohe_features.pkl")  # Save this during training
    return models, preprocessor, ohe_features

models, preprocessor, ohe_features = load_models_and_preprocessor()

# --------------------------
# Streamlit App
# --------------------------
st.title("📊 Customer Churn Prediction")
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", list(models.keys()))

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
# Build Raw Input DataFrame
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
# Ensure all expected OHE columns exist
# --------------------------
for col in ohe_features:
    if col not in input_df.columns:
        input_df[col] = 0  # default value if missing

# --------------------------
# Prediction
# --------------------------
if st.button("Predict"):
    model = models[model_choice]
    try:
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN!")
        else:
            st.success("✅ Customer is NOT likely to churn.")

        if proba is not None:
            st.write("Confidence:", f"{max(proba)*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
