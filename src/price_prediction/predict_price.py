
import joblib
import pandas as pd
from pathlib import Path

# Locate the saved model
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "car_price_pipeline.joblib"

# Load model once
model = joblib.load(MODEL_PATH)


def predict_car_price(car_details):
    """
    Predict used car selling price.

    car_details must contain:
    name, year, km_driven, fuel, seller_type,
    transmission, owner, mileage, engine,
    max_power, seats
    """

    input_data = pd.DataFrame([car_details])

    prediction = model.predict(input_data)[0]

    return round(float(prediction), 2)
