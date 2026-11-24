
import streamlit as st
import pandas as pd
from PIL import Image

from src.damage_detection.detect_damage import detect_damage

# Try importing teammate's price prediction model
try:
    from src.price_prediction.predict_price import predict_car_price
    PRICE_MODEL_AVAILABLE = True
except ImportError:
    PRICE_MODEL_AVAILABLE = False


st.set_page_config(
    page_title="CarWise AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 CarWise AI")
st.subheader("Smart Car Price Prediction & Damage Detection")

st.write(
    "Analyze your car's visible condition and estimate its resale value."
)

st.divider()

# Store results between Streamlit reruns
if "damage_results" not in st.session_state:
    st.session_state.damage_results = []

if "base_price" not in st.session_state:
    st.session_state.base_price = None


# ------------------------------------------------
# SECTION 1: CAR DETAILS AND PRICE PREDICTION
# ------------------------------------------------

st.header("1. Car Price Prediction")

with st.form("car_details_form"):

    col1, col2 = st.columns(2)

    with col1:
        car_name = st.text_input("Car Name", placeholder="e.g. Hyundai Creta")
        yr_mfr = st.number_input("Manufacturing Year", 1990, 2026, 2020)
        fuel_type = st.selectbox(
            "Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
        )
        kms_run = st.number_input("Kilometres Driven", min_value=0)
        city = st.text_input("City", placeholder="e.g. Chandigarh")
        body_type = st.selectbox(
            "Body Type", ["Hatchback", "Sedan", "SUV", "MUV", "Other"]
        )

    with col2:
        transmission = st.selectbox(
            "Transmission", ["Manual", "Automatic"]
        )
        variant = st.text_input("Variant", placeholder="e.g. SX(O)")
        total_owners = st.selectbox(
            "Number of Previous Owners", [1, 2, 3, 4, 5]
        )
        original_price = st.number_input(
            "Original Ex-showroom Price (₹)",
            min_value=0.0
        )
        make = st.text_input("Manufacturer", placeholder="e.g. Hyundai")
        model = st.text_input("Model", placeholder="e.g. Creta")

    submitted = st.form_submit_button("Predict Car Price")

if submitted:

    car_details = {
        "car_name": car_name,
        "yr_mfr": yr_mfr,
        "fuel_type": fuel_type,
        "kms_run": kms_run,
        "city": city,
        "body_type": body_type,
        "transmission": transmission,
        "variant": variant,
        "total_owners": total_owners,
        "original_price": original_price,
        "make": make,
        "model": model
    }

    if PRICE_MODEL_AVAILABLE:
        try:
            st.session_state.base_price = predict_car_price(car_details)
            st.success("Price prediction completed!")
            st.metric(
                "Estimated Base Resale Price",
                f"₹{st.session_state.base_price:,.2f}"
            )
        except Exception as e:
            st.error(f"Price prediction error: {e}")

    else:
        st.warning(
            "Price prediction model is not integrated yet. "
            "It will become available when your teammate's code is added."
        )


st.divider()


# ------------------------------------------------
# SECTION 2: DAMAGE DETECTION
# ------------------------------------------------

st.header("2. Car Damage Detection")

st.write(
    "Upload one or more photographs of your car to identify visible damage."
)

uploaded_images = st.file_uploader(
    "Upload Car Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.button("Analyze Car Damage", disabled=not uploaded_images):

    all_results = []

    progress = st.progress(0)

    for index, uploaded_file in enumerate(uploaded_images):

        image = Image.open(uploaded_file).convert("RGB")

        try:
            result = detect_damage(image)

            all_results.append({
                "filename": uploaded_file.name,
                "original_image": image,
                "annotated_image": result["annotated_image"],
                "detections": result["detections"],
                "damage_counts": result["damage_counts"],
                "total_damages": result["total_damages"]
            })

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

        progress.progress((index + 1) / len(uploaded_images))

    st.session_state.damage_results = all_results

    st.success("Damage analysis completed!")


# Display stored detection results
if st.session_state.damage_results:

    st.subheader("Damage Detection Results")

    for result in st.session_state.damage_results:

        with st.expander(result["filename"], expanded=True):

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    result["original_image"],
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    result["annotated_image"],
                    caption="Detected Damage",
                    use_container_width=True
                )

            st.write("**Detected Damage Details**")

            if result["detections"]:

                df = pd.DataFrame([
                    {
                        "Damage": item["damage"],
                        "Confidence (%)": item["confidence"],
                        "Bounding Box": item["bounding_box"]
                    }
                    for item in result["detections"]
                ])

                st.dataframe(df, use_container_width=True)

                st.write("**Damage Summary**")

                st.json(result["damage_counts"])

            else:
                st.info("No damage detected above the selected confidence threshold.")


st.divider()


# ------------------------------------------------
# SECTION 3: FINAL CONDITION REPORT
# ------------------------------------------------

st.header("3. Final Car Condition Report")

if st.session_state.damage_results:

    total_damages = sum(
        result["total_damages"]
        for result in st.session_state.damage_results
    )

    combined_counts = {}

    for result in st.session_state.damage_results:
        for damage, count in result["damage_counts"].items():
            combined_counts[damage] = (
                combined_counts.get(damage, 0) + count
            )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Images Analyzed", len(st.session_state.damage_results))

    with col2:
        st.metric("Total Damage Detections", total_damages)

    st.write("**Overall Damage Summary**")
    st.json(combined_counts)

    repair_cost = st.number_input(
        "Enter Estimated Repair Cost (₹)",
        min_value=0.0,
        value=0.0,
        key="repair_cost"
    )

    base_price = st.session_state.base_price

    if base_price is not None:

        adjusted_price = max(0, float(base_price) - repair_cost)

        st.metric(
            "Base Resale Price",
            f"₹{float(base_price):,.2f}"
        )

        st.metric(
            "Estimated Repair Cost",
            f"₹{repair_cost:,.2f}"
        )

        st.metric(
            "Adjusted Estimated Price",
            f"₹{adjusted_price:,.2f}"
        )

    else:
        st.info(
            "The damage analysis is ready. "
            "The final adjusted price will appear after price model integration."
        )

    st.caption(
        "Note: The damage detector identifies visible damage only. "
        "It cannot assess hidden structural damage, engine health, "
        "mechanical condition or accident history. Repair costs must "
        "be estimated separately."
    )

else:
    st.info("Upload and analyze car images to generate the condition report.")


st.divider()

st.caption("CarWise AI | Smart Car Price Detection System")
