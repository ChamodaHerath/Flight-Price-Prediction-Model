# Flight Price Prediction Model

This project predicts flight prices based on user-provided inputs using machine learning. It also includes a web-based platform for demonstration.

## Features
The following features are used to predict flight prices:
- **Airline:** The carrier operating the flight.
- **Flight Number:** The unique identifier for a flight.
- **Source and Destination Cities:** The cities where the flight begins and ends.
- **Departure and Arrival Times:** Time categories for the flight's start and end.
- **Number of Stops:** Whether the flight is direct or has connecting stops.
- **Duration:** Total travel time.
- **Class:** Economy or Business class.
- **Days Left to Departure:** The number of days remaining before the flight.

## Machine Learning Models
The model was trained using the following algorithms:
- Random Forest
- XGBoost
- Decision Tree
- CatBoost
- LightGBM

Hyperparameter tuning was performed for the Random Forest model using Randomized Search.

## Web-Based Platform
A web application is provided to demonstrate the model's predictions. Users can input:
- **Airline**
- **Destination City**
- **Arrival City**
- **Class (Economy or Business)**

The app predicts the flight price based on these inputs.

### Prerequisites
To run the web application:
1. Download the trained model file `classifier1.pkl`.
2. Place the `classifier1.pkl` file in the same directory as `app.py`.
   classifier1.pkl: The trained model file (not included in the repository, must be downloaded).



# Thank You!
