from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# Load the saved model
model = load_model('./lstm_aqi_model.keras')

# Load the scaler and the original dataset to fit the scaler
file_path = './los_angeles_data_combined.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data = data.sort_values(by='Date')
data.set_index('Date', inplace=True)
aqi_series = data['AQI']
aqi_series = aqi_series.fillna(method='ffill').fillna(method='bfill')
Q1 = aqi_series.quantile(0.25)
Q3 = aqi_series.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_low = Q1 - 1.5 * IQR
outlier_threshold_high = Q3 + 1.5 * IQR
cleaned_aqi_series = aqi_series[(aqi_series > outlier_threshold_low) & (aqi_series < outlier_threshold_high)]
cleaned_aqi_series = cleaned_aqi_series.asfreq('D')
cleaned_aqi_series = cleaned_aqi_series.fillna(method='ffill').fillna(method='bfill')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cleaned_aqi_series.values.reshape(-1, 1))

# Function to prepare the data
def prepare_data(date_str, look_back=3):
    date = pd.to_datetime(date_str)
    # Get the last available date in the dataset
    last_date = cleaned_aqi_series.index[-1]
    
    if date <= last_date:
        if date not in cleaned_aqi_series.index:
            raise ValueError("Date not found in the dataset. Please provide a valid date.")
        
        # Find the index of the given date
        idx = cleaned_aqi_series.index.get_loc(date)

        if idx < look_back:
            raise ValueError("Not enough data to create a sequence for the provided date.")

        # Prepare the sequence
        sequence = cleaned_aqi_series[idx-look_back:idx].values
        sequence = scaler.transform(sequence.reshape(-1, 1))
        sequence = np.reshape(sequence, (1, look_back, 1))
        prediction = model.predict(sequence)
        prediction = scaler.inverse_transform(prediction)
        predicted_aqi = prediction[0][0]
        return float(predicted_aqi)

    # Prepare the sequence for prediction
    sequence = cleaned_aqi_series[-look_back:].values
    sequence = scaler.transform(sequence.reshape(-1, 1))
    sequence = np.reshape(sequence, (1, look_back, 1))
    
    # Predict iteratively until the target date
    current_date = last_date
    while current_date < date:
        prediction = model.predict(sequence)
        sequence = np.append(sequence[:, 1:, :], np.reshape(prediction, (1, 1, 1)), axis=1)
        current_date += pd.DateOffset(days=1)
    
    # Invert scaling
    predicted_aqi = scaler.inverse_transform(prediction)[0][0]
    return float(predicted_aqi)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        date_str = data['date']
        predicted_aqi = prepare_data(date_str)
        
        return jsonify({'date': date_str, 'predicted_aqi': predicted_aqi})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()