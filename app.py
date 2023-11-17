import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

#loading scaler
with open('standard_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

#loading model
tuned_model = load_model('good_model.h5')


st.title("Customer Churn Prediction Model")

st.sidebar.header("User Input")

# creating  inputs

PaymentMethod_mapping ={'Electronic check':1 ,'Mailed check':1, 'Bank transfer (automatic)':0, 'Credit card (automatic)': 0}
PaymentMethod= st.sidebar.selectbox("PaymentMethod", list(PaymentMethod_mapping.keys()), format_func=lambda x: PaymentMethod_mapping[x])

monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, step=1.0, value=0.0)

PaperlessBilling_mapping = {0: 'No', 1: 'Yes'}
PaperlessBilling = st.sidebar.selectbox("Paperless billing", list(PaperlessBilling_mapping.keys()), format_func=lambda x: PaperlessBilling_mapping[x])

SeniorCitizen_mapping = {0: '0', 1: '1'}
SeniorCitizen = st.sidebar.selectbox("SeniorCitizen", list(SeniorCitizen_mapping.keys()), format_func=lambda x: SeniorCitizen_mapping[x])

StreamingTV_mapping = {0: 'No', 1: 'Yes'}
StreamingTV = st.sidebar.selectbox("StreamingTV", list(StreamingTV_mapping.keys()), format_func=lambda x: StreamingTV_mapping[x])

total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=1.0, value=0.0)
tenure = st.sidebar.number_input("tenure", min_value=0, max_value=150, step=1, value=0)


##['MonthlyCharges', 'PaperlessBilling', 'SeniorCitizen',  'StreamingTV']

#creating a dataframe
if st.sidebar.button("Enter"): 
    user_input = pd.DataFrame({

        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [monthly_charges],
        'PaperlessBilling': [PaperlessBilling],
        'SeniorCitizen': [SeniorCitizen],
        'StreamingTV': [StreamingTV],
        'TotalCharges': [total_charges],
        'tenure': [tenure]
    })

    # scaling the data input (scale it)
    StandardScaler = loaded_scaler
    important_features = ['PaymentMethod','MonthlyCharges', 'PaperlessBilling', 'SeniorCitizen', 'StreamingTV','TotalCharges','tenure']  
    user_input_scaled = StandardScaler.transform(user_input[important_features])
    user_input_scaled_data = pd.DataFrame(user_input_scaled, columns=important_features)

    #  using the tuned model to make predictions
    prediction = tuned_model.predict(user_input_scaled_data)

    # displaying the prediction
    st.subheader("Prediction")
    churn_output = "Churn" if prediction[0, 0] > 0.5 else "No Churn"
    st.write(f"The predicted churn output is: {churn_output}")

    print(f"Predicted Value: {prediction}")

    st.write(f"The predicted value is: {prediction[0, 0]}")