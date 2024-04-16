import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


lstm_model = load_model('main/LSTM_model.h5')
arima_model = pickle.load(open("main/ARIMA_model.pkl", "rb"))
prophet_model = pickle.load(open("main/Prophet_model.pkl", "rb"))
label_encoders = pickle.load(open('main/Label_Encoder.pkl', 'rb'))
minmax_scaler = pickle.load(open('main/Scaler.pkl', 'rb'))


feature_var = json.load(open("main/columns_unique.json"))
data_columns = feature_var['data_columns']
townlist = data_columns['town']
monthlist = data_columns['month']
flat_typelist = data_columns['flat_type']
blocklist = data_columns['block']
street_namelist = data_columns['street_name']
storey_rangelist = data_columns['storey_range']
flat_modellist = data_columns['flat_model']



def preprocess_data(input_df):  
    # One-hot encode categorical columns
    categorical_cols = ['town', 'block', 'street_name', 'flat_model']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, sparse=True)
    
    # Label encode other categorical columns
    for col in ['flat_type', 'storey_range']:
        if col in input_df:
            input_df[col + '_encoded'] = label_encoders[col].transform(input_df[col])
        else:
            st.error(f"Missing column: {col}")
            return None  # Handle missing column case
    
    # Normalize numerical columns
    numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'resale_price', 'year_population']
    if all(column in input_df.columns for column in numerical_cols):
        input_df[numerical_cols] = minmax_scaler.transform(input_df[numerical_cols])
    else:
        missing_cols = [col for col in numerical_cols if col not in input_df.columns]
        st.error(f"Missing numerical columns: {missing_cols}")
        return None  # Handle missing columns case
    
    return input_df



def arima_predict(input_df):
    arima_prediction = arima_model.predict(input_df)
    return arima_prediction
    
def lstm_predict(input_df):
    lstm_prediction = lstm_model.predict(input_df)
    return lstm_prediction
    
def prophet_predict(input_df):
    prophet_prediction = prophet_model.predict(input_df)
    return prophet_prediction

def main():
    st.write("""
    # Singapore House Resale Price Prediction
    
    Predict singapore house resale price with ARIMA, LSTM, and Prophet!
    
    """)

    st.sidebar.header('User Input parameter')
    
    st.sidebar.markdown("""
    [Example CSV Input File](https://raw.githubusercontent.com/Gellstan/Singapore-House-Resale-Price-Prediction/main/main/sample_input_data.csv)                    
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = st.file_uploader("Upload your data", type=["csv"])
        if input_data is not None:
            input_df = pd.read_csv(input_data)
    else:
        def user_input_features():
            month = st.sidebar.select_slider('Month', monthlist)
            town = st.sidebar.selectbox('Town', townlist)
            flat_type = st.sidebar.selectbox('Flat Type', flat_typelist) 
            block = st.sidebar.selectbox('Block', blocklist)
            street_name	= st.sidebar.selectbox('Street Name', street_namelist)
            storey_range = st.sidebar.selectbox('Storey Range', storey_rangelist)
            flat_model = st.sidebar.selectbox('Flat Model', flat_modellist)
            
            have_school_res	= st.sidebar.selectbox('Have School', options=['Yes', 'No'])
            have_school = 1 if have_school_res == 'Yes' else 0
            have_public_transit_res = st.sidebar.selectbox('Have Public Transit', options=['Yes', 'No'])
            have_public_transit = 1 if have_public_transit_res == 'Yes' else 0
            
            floor_area_sqm = st.sidebar.slider('Floor Area(sqm)', 28, 307, 95)
            lease_commence_date = st.sidebar.slider('Lease Commence Date', 1966, 2019, 1986)
            resale_price = st.sidebar.slider('Resale Price', 5000, 1500000, 275000)
            year_population	= st.sidebar.slider('Year Population', 3000000, 6000000, 4250000)
            
            data = {
                'month' : month,
                'town' : town,
                'flat_type'	: flat_type,
                'block'	: block,
                'street_name' : street_name,
                'storey_range' : storey_range,
                'floor_area_sqm' : floor_area_sqm,
                'flat_model' : flat_model,
                'lease_commence_date' : lease_commence_date,
                'resale_price' : resale_price,
                'year_population' : year_population	,
                'have_school' : have_school,
                'have_public_transit' : have_public_transit}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    st.subheader('User Input parameters')
    st.write(input_df)
    st.write('---')
    
    processed_input_df = preprocess_data(input_df)
    arima_prediction = arima_predict(processed_input_df)
    lstm_prediction = lstm_predict(processed_input_df)
    prophet_prediction = prophet_predict(processed_input_df)
    
    st.subheader('ARIMA Prediction')
    st.write(arima_prediction)
    st.write('---')
    
    st.subheader('LSTM Prediction')
    st.write(lstm_prediction)
    st.write('---')
    
    st.subheader('Prophet Prediction')
    st.write(prophet_prediction)
    st.write('---')

main()