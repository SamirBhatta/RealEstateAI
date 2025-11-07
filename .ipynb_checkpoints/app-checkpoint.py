from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")

feature_names = [
    'Bedroom', 'Bathroom', 'Floors', 'Build_Area', 'Road_Width',
    'Amenities_Count', 'Property_Age', 'Distance_mainroad_km',
    'City_Bhaktapur', 'City_Kathmandu', 'City_Lalitpur',
    'Road_Type_Blacktopped', 'Road_Type_Gravelled', 'Road_Type_Soil Stabilized',
    'Furnishing_Fully furnished', 'Furnishing_Semi-furnished', 'Furnishing_Unfurnished',
    'Neighborhood_freq'
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            features = [float(request.form[name]) for name in feature_names]
            features = np.array(features).reshape(1, -1)

            features_scaled = feature_scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            prediction = round(pred, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, feature_names=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
