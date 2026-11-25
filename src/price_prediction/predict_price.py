
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "carwise_random_forest.joblib"

model = joblib.load(MODEL_PATH)

# Dataset listings are primarily from 2024
REFERENCE_YEAR = 2024


def predict_car_price(car_details):

    input_data = pd.DataFrame([{
        "Brand": car_details["Brand"],
        "model": car_details["model"],
        "Year": car_details["Year"],
        "CarAge": REFERENCE_YEAR - car_details["Year"],
        "kmDriven": car_details["kmDriven"],
        "Transmission": car_details["Transmission"],
        "Owner": car_details["Owner"],
        "FuelType": car_details["FuelType"]
    }])

    prediction = model.predict(input_data)[0]

    return round(float(prediction), 2)
