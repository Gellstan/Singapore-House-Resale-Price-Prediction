import streamlit as st
import pandas as pd
import pickle
# from tensorflow.keras.models import load_model
import json
import numpy as np


# lstm_model = load_model('LSTM_model.h5')
# arima_model = pickle.load(open("ARIMA_model.pkl", "rb"))
# prophet_model = pickle.load(open("Prophet_model.pkl", "rb"))

feature_var = json.load(open("columns_unique.json"))
data_columns = feature_var['data_columns']
townlist = data_columns['town']
monthlist = data_columns['month']
flat_typelist = data_columns['flat_type']
blocklist = data_columns['block']
street_namelist = data_columns['street_name']
storey_rangelist = data_columns['storey_range']
flat_modellist = data_columns['flat_model']



def main():
    st.write("""
    # Singapore House Resale Price Prediction
    
    Predict singapore house resale price with ARIMA, LSTM, and Prophet!
    
    """)

    st.sidebar.header('User Input parameter')
    
    st.sidebar.markdown("""
    [Example CSV Input File](the link of github)                    
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            # month = st.sidebar.selectbox('Town', monthlist)
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
            lease_commence_date = st.sidebar.slider('Lease Commence Date', 1966, 2019, 1987)
            resale_price = st.sidebar.slider('Resale Price', 5000, 1500000, 275000)
            year_population	= st.sidebar.slider('Year Population', 3000000, 6000000, 4250000)
            
            data = {
                # 'month' : month,
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



main()