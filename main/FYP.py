import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import numpy as np


lstm_model = load_model('main/LSTM_model.h5')
arima_model = pickle.load(open("main/ARIMA_model.pkl", "rb"))
prophet_model = pickle.load(open("main/Prophet_model.pkl", "rb"))

feature_var = json.load(open("main/columns_unique.json"))
data_columns = feature_var['data_columns']
townlist = data_columns['town']
monthlist = data_columns['month']
flat_typelist = data_columns['flat_type']
blocklist = data_columns['block']
street_namelist = data_columns['street_name']
storey_rangelist = data_columns['storey_range']
flat_modellist = data_columns['flat_model']

def arima_predict(input_df, arima_model):
    # Ensure index is in datetime format
    input_df.index = pd.to_datetime(input_df.index)

    start_date = pd.to_datetime(input_df.index[0])
    end_date = pd.to_datetime(input_df.index[-1])

    trained_start = pd.to_datetime(arima_model.data.row_labels[0])
    trained_end = pd.to_datetime(arima_model.data.row_labels[-1])

    if start_date is None or end_date is None:
        start_date = trained_start
        end_date = trained_end

    if not (trained_start <= start_date <= trained_end and trained_start <= end_date <= trained_end):
        # Handle date out of range
        start_date = max(start_date, trained_start)
        end_date = min(end_date, trained_end)

    # Proceed with prediction
    arima_prediction = arima_model.predict(start=start_date, end=end_date)
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
        input_df = pd.read_csv(uploaded_file)
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
    
    arima_prediction = arima_predict(input_df, arima_model)
    lstm_prediction = lstm_predict(input_df)
    prophet_prediction = prophet_predict(input_df)
    
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