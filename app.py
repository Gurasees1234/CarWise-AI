
import streamlit as st
import pandas as pd
from PIL import Image

from src.damage_detection.detect_damage import detect_damage

try:
    from src.price_prediction.predict_price import predict_car_price
    PRICE_MODEL_AVAILABLE = True
except ImportError as e:
    PRICE_MODEL_AVAILABLE = False
    print("Price model error:", e)


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="CarWise AI | Smart Vehicle Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #172c4d 0%, transparent 42%),
        #0b1220;
    color: #f1f5f9;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #172d50, #10213b 60%, #123b5a);
    border: 1px solid #29496c;
    border-radius: 22px;
    padding: 36px;
    margin-bottom: 30px;
}

.hero-tag {
    color: #60c7ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0 8px 0;
}

.hero p {
    color: #c4d3e5;
    font-size: 15px;
    margin-bottom: 0;
}

.section-heading {
    font-size: 23px;
    font-weight: 700;
    color: #f8fafc;
    margin: 12px 0 5px 0;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 22px;
}

.info-card {
    background: linear-gradient(135deg, #17263b, #111d30);
    border: 1px solid #2a405c;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0;
}

.price-card {
    background: linear-gradient(120deg, #123c56, #142c4c);
    border: 1px solid #2875a0;
    border-radius: 18px;
    padding: 26px;
    margin-top: 22px;
}

.price-label {
    color: #a8cce5;
    font-size: 13px;
    font-weight: 600;
}

.price-value {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}

.price-note {
    color: #a8cce5;
    font-size: 12px;
}

div[data-testid="stForm"] {
    background: #111d30;
    border: 1px solid #2b405c;
    border-radius: 18px;
    padding: 22px;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: #1a2a40;
    border: 1px solid #354a64;
    border-radius: 9px;
    color: white;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #1a2a40;
    border-color: #354a64;
    border-radius: 9px;
}

label {
    color: #dbeafe !important;
    font-weight: 500 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #1677d2, #259be8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(100deg, #258fe8, #45b3ff);
    border: none;
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: #111d30;
    border: 1px solid #263b55;
    padding: 9px;
    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #aabbd0;
    padding: 12px 20px;
}

.stTabs [aria-selected="true"] {
    background: #1b4c78 !important;
    color: white !important;
}

div[data-testid="stFileUploader"] {
    background: #111d30;
    border: 1px dashed #426384;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stMetric"] {
    background: #14253b;
    border: 1px solid #2d4968;
    border-radius: 13px;
    padding: 17px;
}

hr {
    border-color: #293d56;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "damage_results" not in st.session_state:
    st.session_state.damage_results = []

if "base_price" not in st.session_state:
    st.session_state.base_price = None


# ------------------------------------------------
# HERO SECTION
# ------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI Powered Automotive Intelligence</div>
    <h1>🚘 CarWise AI</h1>
    <p>
        Smarter car valuations. Intelligent damage detection.
        Make better decisions when buying or selling used cars.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------
# NAVIGATION TABS
# ------------------------------------------------

price_tab, damage_tab, report_tab = st.tabs([
    "💰  Price Estimator",
    "🔍  Damage Inspection",
    "📊  Final Valuation"
])


# =================================================
# TAB 1: PRICE PREDICTION
# =================================================

with price_tab:

    st.markdown(
        '<div class="section-heading">Used Car Price Estimator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter your vehicle specifications to estimate its resale value.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("car_details_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("##### 🚗 Vehicle Information")

            car_name = st.text_input(
                "Car Name",
                placeholder="e.g. Maruti Swift Dzire VDI"
            )

            year = st.number_input(
                "Manufacturing Year",
                min_value=1990,
                max_value=2026,
                value=2018
            )

            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                value=45000
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["Diesel", "Petrol", "LPG", "CNG"]
            )

            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"]
            )

            seller_type = st.selectbox(
                "Seller Type",
                ["Individual", "Dealer", "Trustmark Dealer"]
            )

        with col2:

            st.markdown("##### ⚙️ Technical Specifications")

            owner = st.selectbox(
                "Previous Owners",
                [
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner",
                    "Test Drive Car"
                ]
            )

            mileage = st.number_input(
                "Mileage (kmpl)",
                min_value=0.0,
                value=18.0,
                step=0.1
            )

            engine = st.number_input(
                "Engine Capacity (CC)",
                min_value=0,
                value=1197
            )

            max_power = st.number_input(
                "Maximum Power (bhp)",
                min_value=0.0,
                value=82.0,
                step=0.1
            )

            seats = st.selectbox(
                "Number of Seats",
                [2, 4, 5, 6, 7, 8, 9, 10, 14]
            )

        submitted = st.form_submit_button(
            "✨  Calculate Estimated Price",
            use_container_width=True
        )

    if submitted:

        if not car_name.strip():
            st.warning("Please enter the car name.")

        elif not PRICE_MODEL_AVAILABLE:
            st.error("Price prediction model is unavailable.")

        else:

            car_details = {
                "name": car_name.strip(),
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats
            }

            try:

                predicted_price = predict_car_price(car_details)

                st.session_state.base_price = predicted_price

                st.success("Vehicle valuation generated successfully!")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    if st.session_state.base_price is not None:

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">
                    YOUR AI-BASED INDICATIVE MARKET VALUATION
                </div>
                <div class="price-value">
                    ₹{st.session_state.base_price:,.0f}
                </div>
                <div class="price-note">
                    Based on historical used-car listing data.
                    Actual selling prices may vary.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Note: This is an indicative estimate, not a guaranteed "
            "current market or transaction price."
        )


# =================================================
# TAB 2: DAMAGE DETECTION
# =================================================

with damage_tab:

    st.markdown(
        '<div class="section-heading">Vehicle Damage Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload vehicle photographs to identify visible exterior damage.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_images = st.file_uploader(
        "Upload vehicle photographs",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="damage_upload"
    )

    if uploaded_images:

        st.caption(f"{len(uploaded_images)} image(s) selected.")

    if st.button(
        "🔍  Analyze Vehicle Damage",
        disabled=not uploaded_images,
        use_container_width=True
    ):

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
                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

            progress.progress((index + 1) / len(uploaded_images))

        st.session_state.damage_results = all_results

        st.success("Damage inspection completed!")

    if st.session_state.damage_results:

        st.markdown("### Inspection Results")

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

                st.markdown("#### Damage Details")

                if result["detections"]:

                    damage_df = pd.DataFrame([
                        {
                            "Damage": item["damage"],
                            "Confidence (%)": item["confidence"],
                            "Bounding Box": item["bounding_box"]
                        }
                        for item in result["detections"]
                    ])

                    st.dataframe(
                        damage_df,
                        use_container_width=True
                    )

                    st.json(result["damage_counts"])

                else:
                    st.info(
                        "No damage detected above the selected "
                        "confidence threshold."
                    )


# =================================================
# TAB 3: FINAL CONDITION REPORT
# =================================================

with report_tab:

    st.markdown(
        '<div class="section-heading">Final Vehicle Valuation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Review the estimated resale value and visible damage findings.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.base_price is not None:

        st.metric(
            "AI-Based Base Resale Estimate",
            f"₹{st.session_state.base_price:,.0f}"
        )

    else:

        st.info(
            "Generate a car price prediction first to view the valuation."
        )

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

        col1.metric(
            "Images Analyzed",
            len(st.session_state.damage_results)
        )

        col2.metric(
            "Total Damage Detections",
            total_damages
        )

        st.markdown("### Damage Summary")

        st.json(combined_counts)

        st.markdown("### Repair Cost Adjustment")

        repair_cost = st.number_input(
            "Estimated Repair Cost (₹)",
            min_value=0.0,
            value=0.0,
            key="repair_cost"
        )

        if st.session_state.base_price is not None:

            base_price = float(st.session_state.base_price)

            adjusted_price = max(0, base_price - repair_cost)

            col1, col2 = st.columns(2)

            col1.metric(
                "Base Resale Estimate",
                f"₹{base_price:,.0f}"
            )

            col2.metric(
                "Estimated Repair Cost",
                f"₹{repair_cost:,.0f}"
            )

            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">
                        ADJUSTED INDICATIVE VALUATION
                    </div>
                    <div class="price-value">
                        ₹{adjusted_price:,.0f}
                    </div>
                    <div class="price-note">
                        Base estimate minus user-entered repair cost.
                        This is a simplified adjustment, not a market appraisal.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "Complete a vehicle damage inspection to generate "
            "the full condition report."
        )

    st.caption(
        "CarWise AI identifies visible damage only. It cannot assess "
        "hidden structural damage, engine health, mechanical condition "
        "or accident history. Repair costs are entered separately."
    )


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8193aa; padding:15px;">
        <b style="color:#c7d8ed;">🚘 CarWise AI</b><br>
        Smart Car Price Prediction & Damage Detection<br>
        <small>AI-powered insights for smarter vehicle decisions.</small>
    </div>
    """,
    unsafe_allow_html=True
)
