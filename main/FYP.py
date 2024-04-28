import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# lstm_model = load_model('main/LSTM_model.h5')
arima_model = pickle.load(open("main/ARIMA_model.pkl", "rb"))
prophet_model = pickle.load(open("main/Prophet_model.pkl", "rb"))
label_encoders_flat_type = pickle.load(open('main/Label_Encoder_Flat_Type.pkl', 'rb'))
label_encoders_storey_range = pickle.load(open('main/Label_Encoder_Storey_Range.pkl', 'rb'))
minmax_scaler = pickle.load(open('main/Scaler.pkl', 'rb'))
price_scaler = pickle.load(open('main/Scaler_ARIMA.pkl', 'rb'))

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
    input_df['flat_type_encoded'] = label_encoders_flat_type.transform(input_df['flat_type'])
    input_df['storey_range_encoded'] = label_encoders_storey_range.transform(input_df['storey_range'])

    # Normalize numerical columns
    numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'resale_price', 'year_population']
    input_df[numerical_cols] = minmax_scaler.transform(input_df[numerical_cols])
    
    return input_df

def arima_invert_scaling(scaled_predictions):
    last_known_value = 0.3629080527602271
    reintegrated_forecast = np.cumsum(np.insert(scaled_predictions, 0, last_known_value))[-24:]
    scaled_predictions = np.array(reintegrated_forecast).reshape(-1, 1)
    original_scale_predictions = price_scaler.inverse_transform(scaled_predictions)
    return original_scale_predictions

def arima_predict(input_df):
    # Check if 'month' is a column, convert it to datetime, and set it as the index
    if 'month' in input_df.columns:
        input_df.set_index('month', inplace=True)
    
    # Extract the 'resale_price' series
    if 'resale_price' in input_df.columns:
        input_series = input_df['resale_price']
    else:
        raise ValueError("resale_price column is missing from the input DataFrame.")

    # Ensure that the index is properly formatted as datetime
    if not isinstance(input_series.index, pd.DatetimeIndex):
        raise TypeError("Index is not a datetime type, which is required for ARIMA predictions.")

    # Perform prediction using the ARIMA model
    end = '2030-12'
    arima_prediction = arima_model.predict(start=input_series.index, end=end)
    return arima_prediction
    
def prophet_invert_scaling(scaled_predictions):
    if isinstance(scaled_predictions, pd.DataFrame):
        numeric_cols = scaled_predictions.select_dtypes(include=[np.number])
        scaled_array = numeric_cols.values
    else:
        scaled_array = np.array(scaled_predictions).reshape(-1, 1)

        # Assuming scaled_predictions is an array of numeric values
    original_scale_predictions = price_scaler.inverse_transform(scaled_array)
    original_scale_predictions = original_scale_predictions.astype(float)
    # Convert array back to DataFrame and add necessary columns
    result_df = pd.DataFrame(original_scale_predictions, columns=['predicted_value','predicted_value_lower','predicted_value_upper'])
    start_date = scaled_predictions['ds'].min()
    result_df['ds'] = pd.date_range(start=start_date, periods=result_df.shape[0], freq='M')
    result_df.set_index('ds',inplace=True)
    return result_df
    
def prophet_predict(input_df, start_price):
    # Ensure input_df has 'month' in datetime format and set as index
    if not pd.api.types.is_datetime64_any_dtype(input_df['month']):
        input_df['month'] = pd.to_datetime(input_df['month'])
    input_df.set_index('month', inplace=True)

    # Assume the starting date and price are given by the user and simulate this as past data
    prophet_df = pd.DataFrame({
        'ds': input_df.index,
        'y': np.full(len(input_df.index), start_price)  # Set all entries to start_price
    })

    # Future dataframe creation extending to 2030-12
    future = prophet_model.make_future_dataframe(periods=(2030 - prophet_df['ds'].dt.year.max()) * 12 + 12, freq='M')
    future = pd.concat([prophet_df, future[future['ds'] > prophet_df['ds'].max()]])

    # Prediction
    forecast = prophet_model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]



def predicted_plot(unscaled_prophet_prediction):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure the index is a datetime type without timezone issues
    if unscaled_prophet_prediction.index.tz is not None:
        unscaled_prophet_prediction.index = unscaled_prophet_prediction.index.tz_localize(None)
    
    # Convert dates explicitly to matplotlib's internal representation of dates
    dates = mdates.date2num(unscaled_prophet_prediction.index.to_pydatetime())
    ax.plot(dates, unscaled_prophet_prediction['predicted_value'], label='Predicted', color='orange')
    ax.fill_between(dates, 
                    unscaled_prophet_prediction['predicted_value_lower'], 
                    unscaled_prophet_prediction['predicted_value_upper'], 
                    color='gray', alpha=0.2, label='Confidence Interval')
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()

    st.pyplot(fig)


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
                'town': town,
                'flat_type': flat_type,
                'block': block,
                'street_name': street_name,
                'storey_range': storey_range,
                'floor_area_sqm': float(floor_area_sqm),
                'flat_model': flat_model,
                'lease_commence_date': int(lease_commence_date),
                'resale_price': float(resale_price),
                'year_population': int(year_population),
                'have_school': int(have_school),
                'have_public_transit': int(have_public_transit)
            }
            features = pd.DataFrame(data, index=[0])
            features['month'] = pd.to_datetime(features['month'], errors='coerce')
            return features
        input_df = user_input_features()

    st.subheader('User Input parameters')
    st.write(input_df)
    st.write('---')
    
    st.subheader('Prophet Prediction')
    processed_input_df = preprocess_data(input_df)
    prophet_prediction = prophet_predict(processed_input_df)
    unscaled_prophet_prediction = prophet_invert_scaling(prophet_prediction)
    st.write(unscaled_prophet_prediction)
    predicted_plot(unscaled_prophet_prediction)
    st.write('---')
    
    #st.subheader('ARIMA Prediction')
    #arima_prediction = arima_predict(processed_input_df)
    #unscaled_arima_prediction = arima_invert_scaling(arima_prediction)
    #st.write(unscaled_arima_prediction)
    #Graph
    #st.write('---')

main()