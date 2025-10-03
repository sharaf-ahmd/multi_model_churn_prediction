import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load Saved Models
# --------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("gnb_model.pkl"),
    }
    return models

models = load_models()

# --------------------------
# Load Preprocessor
# --------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

preprocessor = load_preprocessor()

# --------------------------
# Streamlit App
# --------------------------
st.title("📊 Customer Churn Prediction")

# Sidebar: choose model
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", list(models.keys()))

# --------------------------
# Collect User Inputs
# --------------------------
st.header("Enter Customer Information")

quantity = st.number_input("Quantity", min_value=0, value=1)
returns = st.number_input("Returns", min_value=0, value=0)
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
# Build Input DataFrame
# --------------------------
# Columns must match preprocessor training
all_columns = [
    "Quantity", "Returns", "Gender",
    "purchase_year", "purchase_month", "purchase_day",
    "Product Category_Books", "Product Category_Clothing", "Product Category_Electronics", "Product Category_Home",
    "Payment Method_Credit Card", "Payment Method_PayPal", "Payment Method_Cash",
    "0-19", "20-29", "30-39", "40-49", "50-59", "60+",
    "Total Purchase Amount_scaled", "Product Price_scaled"
]

# Initialize all columns to 0
input_dict = {col: 0 for col in all_columns}

# Fill numeric
input_dict["Quantity"] = quantity
input_dict["Returns"] = returns
input_dict["Gender"] = 1 if gender == "Male" else 0
input_dict["purchase_year"] = purchase_year
input_dict["purchase_month"] = purchase_month
input_dict["purchase_day"] = purchase_day
input_dict["Total Purchase Amount_scaled"] = total_purchase
input_dict["Product Price_scaled"] = product_price

# One-hot encode product category
if f"Product Category_{product_category}" in input_dict:
    input_dict[f"Product Category_{product_category}"] = 1

# One-hot encode payment method
if payment_method == "Credit Card":
    input_dict["Payment Method_Credit Card"] = 1
elif payment_method == "PayPal":
    input_dict["Payment Method_PayPal"] = 1
elif payment_method == "Cash":
    input_dict["Payment Method_Cash"] = 1
# If "Other", all remain 0

# One-hot encode age group
if age_group in input_dict:
    input_dict[age_group] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# --------------------------
# Make Prediction
# --------------------------
if st.button("Predict"):
    model = models[model_choice]

    # Preprocess input
    X_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0] if hasattr(model, "predict_proba") else None

    # Display
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is NOT likely to churn.")

    if proba is not None:
        st.write("Confidence:", f"{max(proba)*100:.2f}%")
