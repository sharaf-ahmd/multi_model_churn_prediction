import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load Saved Models & Preprocessor
# --------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("gnb_model.pkl"),
    }
    return models

@st.cache_resource
def load_preprocessor():
    preprocessor = joblib.load("preprocessor.pkl")
    return preprocessor

models = load_models()
preprocessor = load_preprocessor()

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
# Build Input Data
# --------------------------
# Start with all columns expected by the preprocessor
all_columns = preprocessor.feature_names_in_  # fetch expected columns from preprocessor
input_dict = {col: 0 for col in all_columns}

# Fill numeric features
input_dict["Quantity"] = quantity
input_dict["Returns"] = returns
input_dict["Gender"] = 1 if gender == "Male" else 0
input_dict["purchase_year"] = purchase_year
input_dict["purchase_month"] = purchase_month
input_dict["purchase_day"] = purchase_day
input_dict["Total Purchase Amount_scaled"] = total_purchase
input_dict["Product Price_scaled"] = product_price

# One-hot encode product category
cat_col = f"Product Category_{product_category}"
if cat_col in input_dict:
    input_dict[cat_col] = 1

# One-hot encode payment method
pay_col = f"Payment Method_{payment_method}"
if pay_col in input_dict:
    input_dict[pay_col] = 1

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

    # Apply preprocessor first if using pipeline-less model
    try:
        X_processed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is NOT likely to churn.")

    if proba is not None:
        st.write("Confidence:", f"{max(proba)*100:.2f}%")
