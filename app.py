from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Create LabelEncoder and MinMaxScaler objects
label_encoders = {
    'airline': LabelEncoder(),
    'source_city': LabelEncoder(),
    'destination_city': LabelEncoder(),
    'class': LabelEncoder(),
    'stops': LabelEncoder(),
    'departure_time': LabelEncoder(),
    'arrival_time': LabelEncoder()
}
scaler = MinMaxScaler(feature_range=(0, 1))

# Load your saved Random Forest model
random_forest_model = joblib.load('Random_Forest_regression.joblib')

# Fit LabelEncoders with all possible categories from the training data
# Replace these with the actual categories from your training data
all_categories = {
    'airline': ['Spicejet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_india'],
    'source_city': ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
    'destination_city': ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
    'class': ['Economy', 'Business'],
    'stops': ['Zero', 'One Stops', 'two_or_more'],
    'departure_time': ['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night', 'Late_Night'],
    'arrival_time': ['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night', 'Late_Night']
}

for col in label_encoders:
    label_encoders[col].fit(all_categories[col])

# Define the home route
@app.route('/')
def home():
    return render_template('index.html',
                           airline_options=all_categories['airline'],
                           source_city_options=all_categories['source_city'],
                           destination_city_options=all_categories['destination_city'],
                           class_options=all_categories['class'],
                           departure_time_options=all_categories['departure_time'],
                           stops_options=all_categories['stops'])

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    selected_airline = request.form['airline']
    selected_source_city = request.form['source_city']
    selected_destination_city = request.form['destination_city']
    selected_class = request.form['class']
    selected_departure_time = request.form['departure_time']
    selected_stops = request.form['stops']

    input_data = pd.DataFrame([{
        'airline': selected_airline,
        'source_city': selected_source_city,
        'destination_city': selected_destination_city,
        'class': selected_class,
        'departure_time': selected_departure_time,
        'stops': selected_stops,
        'arrival_time': 'Evening',
        'duration': 3.5,
        'days_left': 10,
        'price': 0,
        'Unnamed: 0' : 1
    }])

    # Encode categorical features
    for col in input_data.columns:
        if col in label_encoders and input_data[col].dtype == 'object':
            le = label_encoders[col]
            input_data[col] = le.transform([input_data[col][0]])

    # Scale the numerical variables using MinMaxScaler
    input_features_scaled = scaler.fit_transform(input_data)

    # Make prediction using the Random Forest model
    predicted_price = random_forest_model.predict(input_features_scaled)

    return render_template('predict.html',
                           prediction=f'Predicted Price: {predicted_price[0]:.2f}',
                           airline_options=all_categories['airline'],
                           source_city_options=all_categories['source_city'],
                           destination_city_options=all_categories['destination_city'],
                           class_options=all_categories['class'],
                           departure_time_options=all_categories['departure_time'],
                           stops_options=all_categories['stops'])

if __name__ == '__main__':
    app.run(debug=True)
