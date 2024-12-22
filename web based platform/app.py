
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import random

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('classifier1.pkl')

# Define the valid options for each feature
VALID_OPTIONS = {
    'airline': ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'],
    'source_city': ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
    'destination_city': ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
    'class': ['Economy', 'Business']
}

# Dummy flight details data (you'd replace this with actual data source)
FLIGHT_DETAILS = {
    ('Delhi', 'Mumbai'): [
        {
            'airline': 'Air India',
            'flight_number': 'AI-101',
            'departure_time': '07:30 AM',
            'arrival_time': '09:30 AM',
            'duration': '2h 00m',
            'stops': 'Non-stop',
            'economy_price': 4500,
            'business_price': 9000
        },
        {
            'airline': 'Vistara',
            'flight_number': 'UK-201',
            'departure_time': '10:15 AM',
            'arrival_time': '12:15 PM',
            'duration': '2h 00m',
            'stops': 'Non-stop',
            'economy_price': 4800,
            'business_price': 9500
        },
        {
            'airline': 'IndiGo',
            'flight_number': '6E-301',
            'departure_time': '02:45 PM',
            'arrival_time': '04:45 PM',
            'duration': '2h 00m',
            'stops': 'Non-stop',
            'economy_price': 4200,
            'business_price': 8500
        }
    ],
    # Add more route-specific flight details as needed
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        features = {
            'airline': request.form['airline'],
            'source_city': request.form['source_city'],
            'destination_city': request.form['destination_city'],
            'class': request.form['class']
        }

        # Check if any values are None or empty
        if any(not value for value in features.values()):
            return render_template('index.html', 
                                error="Please fill all fields",
                                selected=features)

        # Validate inputs
        for feature, value in features.items():
            if value not in VALID_OPTIONS[feature]:
                return jsonify({'error': f'Invalid value for {feature}'})
            
        if features['source_city'] == features['destination_city']:
            return render_template('index.html', error="Source and destination cities cannot be the same")

        # Create one-hot encoded features
        input_data = pd.DataFrame([features])
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_data, columns=['airline', 'source_city', 'destination_city', 'class'])
        
        expected_columns = model.feature_names_in_ 
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match training data
        input_encoded = input_encoded[expected_columns]

        # Make prediction
        prediction = model.predict(input_encoded)[0]

        # Format prediction as currency
        formatted_prediction = f"{prediction:,.2f}"

        # Get flight details for the route
        route_key = (features['source_city'], features['destination_city'])
        flight_details = FLIGHT_DETAILS.get(route_key, [])

        # Filter flight details based on selected class
        filtered_flights = [
            flight for flight in flight_details 
            if (features['class'] == 'Economy' and flight['economy_price']) or 
               (features['class'] == 'Business' and flight['business_price'])
        ]

        # Return result to the user
        return render_template('index.html', 
                             prediction=formatted_prediction,
                             selected=features,
                             show_prediction=True,
                             flight_details=filtered_flights,
                             selected_class=features['class'])

    except Exception as e:
        print(f"Debug: Error occurred - {str(e)}")  # Debug print
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}",
                             selected=features)

if __name__ == "__main__":
    app.run(debug=True)