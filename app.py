import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ===== Load model and preprocessing objects =====
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('feature_order.pkl', 'rb') as file:
    feature_order = pickle.load(file)

# ===== App title =====
st.title('📊 Customer Churn Prediction')

# ===== Sidebar threshold =====
threshold = st.sidebar.slider(
    'Prediction Threshold', min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Adjust the probability cutoff for classifying churn"
)

# ===== User inputs =====
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, step=1)
age = st.slider('🎂 Age', 18, 92)
tenure = st.slider('📅 Tenure (Years)', 0, 10)
balance = st.number_input('🏦 Balance', min_value=0.0, step=100.0)
number_of_products = st.slider('🛒 Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'])
is_active_member = st.selectbox('✅ Is Active Member', ['No', 'Yes'])
estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, step=100.0)

# ===== Prepare input dataframe =====
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded_df = pd.DataFrame(
    onehot_encoder_geo.transform([[geography]]).toarray(),
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data[feature_order]
input_data_scaled = scaler.transform(input_data)

# ===== Prediction =====
prediction_proba = float(model.predict(input_data_scaled)[0])
prediction_percent = prediction_proba * 100

# ===== Decision =====
if prediction_proba >= threshold:
    st.error(f"🚨 Likely to Churn — **{prediction_percent:.2f}% probability**")
    # Simple recommendation logic
    recommendations = []
    if credit_score < 500:
        recommendations.append("Offer a credit score improvement program.")
    if tenure < 3:
        recommendations.append("Consider a loyalty reward to increase retention.")
    if balance < 1000:
        recommendations.append("Provide a balance bonus or savings incentive.")
    if number_of_products == 1:
        recommendations.append("Promote cross-selling of additional products.")
    if is_active_member == 'No':
        recommendations.append("Engage with targeted offers to increase activity.")
    
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(f"- {rec}")
else:
    st.success(f"✅ Not Likely to Churn — **{prediction_percent:.2f}% probability**")
    st.write("Customer profile appears stable. Continue current engagement strategy.")
