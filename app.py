import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load Saved Models
# --------------------------
@st.cache_resource
def load_models():
    # Corrected to load only the models that were actually saved in the notebook.
    # 'svc_model.pkl' was never created, so it's removed.
    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("gnb_model.pkl"),
    }
    return models

models = load_models()

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", list(models.keys()))

# --------------------------
# Collect User Inputs
# --------------------------
st.header("Enter Customer Information")

# Create columns for a better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Purchase Details")
    product_category = st.selectbox("Product Category", ["Books", "Clothing", "Electronics", "Home"])
    quantity = st.number_input("Quantity", min_value=1, value=1)
    returns = st.selectbox("Has the item been returned?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    product_price = st.number_input("Product Price", min_value=10.0, max_value=500.0, value=250.0)
    total_purchase = st.number_input("Total Purchase Amount", min_value=100.0, max_value=5350.0, value=2700.0)

with col2:
    st.subheader("Customer Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_age = st.number_input("Customer Age", min_value=18, max_value=70, value=35)
    
with col3:
    st.subheader("Transaction Info")
    payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash"])
    purchase_year = st.number_input("Purchase Year", min_value=2020, max_value=2023, value=2023)
    purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, value=5)
    purchase_day = st.selectbox("Purchase Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])


# --------------------------
# Prediction Logic
# --------------------------
if st.button("Predict Churn"):
    # The saved pipeline expects a DataFrame that has been manually encoded first, 
    # and then it applies a StandardScaler. We must replicate the manual steps exactly.

    # 1. Create a dictionary with all possible feature columns, initialized to 0 or False
    input_data = {
        'Quantity': 0, 'Returns': 0.0, 'Gender': 0, 'purchase_year': 0,
        'purchase_month': 0, 'purchase_day': 0,
        'Product Category_Books': False, 'Product Category_Clothing': False,
        'Product Category_Electronics': False, 'Product Category_Home': False,
        'Payment Method_Cash': False, 'Payment Method_Credit Card': False,
        'Payment Method_PayPal': False, '0-19': False, '20-29': False,
        '30-39': False, '40-49': False, '50-59': False, '60+': False,
        'Total Purchase Amount_scaled': 0.0, 'Product Price_scaled': 0.0
    }

    # 2. Populate the dictionary with user input, applying the same logic as the notebook
    
    # Simple assignments
    input_data['Quantity'] = quantity
    input_data['Returns'] = float(returns)
    input_data['Gender'] = 1 if gender == "Male" else 0
    input_data['purchase_year'] = purchase_year
    input_data['purchase_month'] = purchase_month
    input_data['purchase_day'] = purchase_day
    
    # One-hot encoding for Product Category
    input_data[f'Product Category_{product_category}'] = True
    
    # One-hot encoding for Payment Method
    input_data[f'Payment Method_{payment_method}'] = True
    
    # Binning and one-hot encoding for Age
    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60+']
    age_group = pd.cut([customer_age], bins=age_bins, labels=age_labels, right=True)[0]
    input_data[age_group] = True

    # Scaling for purchase amounts (using Min-Max scaling as done in notebook)
    # Min/Max values are taken from your notebook's data.describe() output
    input_data['Total Purchase Amount_scaled'] = (total_purchase - 100) / (5350 - 100)
    input_data['Product Price_scaled'] = (product_price - 10) / (500 - 10)

    # 3. Create DataFrame and ensure column order is correct
    input_df = pd.DataFrame([input_data])
    
    # This is the exact order of columns your pipeline was trained on
    expected_columns = [
        'Quantity', 'Returns', 'Gender', 'purchase_year', 'purchase_month',
        'purchase_day', 'Product Category_Books', 'Product Category_Clothing',
        'Product Category_Electronics', 'Product Category_Home',
        'Payment Method_Cash', 'Payment Method_Credit Card',
        'Payment Method_PayPal', '0-19', '20-29', '30-39', '40-49', '50-59',
        '60+', 'Total Purchase Amount_scaled', 'Product Price_scaled'
    ]
    input_df = input_df[expected_columns]

    # 4. Make Prediction
    model = models[model_choice]
    
    # The loaded model is a pipeline that includes the final StandardScaler step
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("⚠ *Prediction: Customer is likely to CHURN!*")
    else:
        st.success("✅ *Prediction: Customer is likely to continue.*")

    if proba is not None:
        confidence = max(proba) * 100
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
