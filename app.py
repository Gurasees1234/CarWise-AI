
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "used_cars_dataset_v2.csv"


# ------------------------------------------------
# LOAD DATASET OPTIONS
# ------------------------------------------------

@st.cache_data
def load_car_options():
    data = pd.read_csv(DATASET_PATH)

    brands = sorted(data["Brand"].dropna().astype(str).unique())

    fuel_types = sorted(
        data["FuelType"].dropna().astype(str).unique()
    )

    transmissions = sorted(
        data["Transmission"].dropna().astype(str).unique()
    )

    owners = sorted(
        data["Owner"].dropna().astype(str).unique()
    )

    return data, brands, fuel_types, transmissions, owners


try:
    car_data, brands, fuel_types, transmissions, owners = load_car_options()
    DATA_OPTIONS_AVAILABLE = True
except Exception as e:
    DATA_OPTIONS_AVAILABLE = False
    print("Dataset options error:", e)


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

DEFAULT_STATE = {
    "workflow_step": 1,
    "damage_results": [],
    "base_price": None,
    "car_details": None,
    "damage_analysis_done": False,
    "exterior_repair_cost_input": 0.0,
    "condition_ratings": {}
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value




# ------------------------------------------------
# VALUATION ESTIMATION RULES
# ------------------------------------------------

# Illustrative repair estimates in INR per detected box.
# These are editable estimates, not verified repair quotations.
DAMAGE_REPAIR_ESTIMATES = {
    "dent": 8000,
    "scratch": 3000,
    "crack": 7000,
    "broken_light": 5000,
    "broken_glass": 12000
}

DEFAULT_DAMAGE_REPAIR_ESTIMATE = 5000

# Prototype condition adjustments.
# These are heuristic percentages of the base predicted price.
CONDITION_ADJUSTMENTS = {
    "Engine": {
        "Good": 0.00,
        "Average": 0.03,
        "Poor": 0.08
    },
    "Transmission": {
        "Good": 0.00,
        "Average": 0.02,
        "Poor": 0.05
    },
    "Brakes": {
        "Good": 0.00,
        "Average": 0.01,
        "Poor": 0.02
    },
    "Tyres": {
        "Good": 0.00,
        "Average": 0.01,
        "Poor": 0.02
    }
}


def estimate_exterior_repair_cost(damage_counts):
    total = 0

    for damage_type, count in damage_counts.items():
        damage_key = str(damage_type).lower().replace(" ", "_")

        rate = DAMAGE_REPAIR_ESTIMATES.get(
            damage_key,
            DEFAULT_DAMAGE_REPAIR_ESTIMATE
        )

        total += rate * int(count)

    return total


def calculate_mechanical_deduction(base_price, ratings):
    breakdown = {}

    for component, condition in ratings.items():
        percentage = CONDITION_ADJUSTMENTS[component][condition]
        deduction = base_price * percentage

        breakdown[component] = {
            "condition": condition,
            "percentage": percentage * 100,
            "deduction": deduction
        }

    total = sum(
        item["deduction"] for item in breakdown.values()
    )

    return total, breakdown



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



# =================================================
# CARWISE AI - STEP-BY-STEP VALUATION WIZARD
# =================================================

st.markdown(
    '<div class="section-heading">Complete Vehicle Valuation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Follow the steps to generate your vehicle condition report.'
    '</div>',
    unsafe_allow_html=True
)

# Automatically scroll to the top when navigating between steps
if st.session_state.get("scroll_to_top", False):
    st.markdown(
        """
        <script>
            window.parent.document.querySelector(
                'section.main'
            ).scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        </script>
        """,
        unsafe_allow_html=True
    )
    st.session_state.scroll_to_top = False

step = st.session_state.workflow_step

# Scroll to top after navigating to another step
if st.session_state.get("scroll_to_top", False):

    components.html(
        """
        <script>
            function scrollPageToTop() {
                try {
                    const parentWindow = window.parent;
                    const parentDocument = parentWindow.document;

                    // Scroll the browser window
                    parentWindow.scrollTo({
                        top: 0,
                        behavior: "smooth"
                    });

                    // Scroll Streamlit's main content container
                    const main = parentDocument.querySelector("section.main");

                    if (main) {
                        main.scrollTo({
                            top: 0,
                            behavior: "smooth"
                        });
                    }

                    const appView = parentDocument.querySelector(
                        '[data-testid="stAppViewContainer"]'
                    );

                    if (appView) {
                        appView.scrollTo({
                            top: 0,
                            behavior: "smooth"
                        });
                    }

                } catch (error) {
                    console.error("Scroll error:", error);
                }
            }

            setTimeout(scrollPageToTop, 150);
        </script>
        """,
        height=0
    )

    st.session_state.scroll_to_top = False

step_names = [
    "Vehicle Details",
    "Exterior Inspection",
    "Mechanical Condition",
    "Final Valuation"
]

st.progress(step / 4)
st.caption(f"Step {step} of 4 — {step_names[step - 1]}")


# =================================================
# STEP 1: VEHICLE DETAILS
# =================================================

if step == 1:

    st.markdown("### 🚘 Step 1: Vehicle Information")

    if not DATA_OPTIONS_AVAILABLE:
        st.error("Could not load the vehicle dataset.")

    elif not PRICE_MODEL_AVAILABLE:
        st.error("Price prediction model is unavailable.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Vehicle Information")

            brand = st.selectbox(
                "Manufacturer",
                brands,
                key="wizard_brand"
            )

            available_models = sorted(
                car_data.loc[
                    car_data["Brand"].astype(str).str.strip().str.casefold()
                    == brand.strip().casefold(),
                    "model"
                ].dropna().astype(str).str.strip().unique()
            )

            if available_models:
                car_model = st.selectbox(
                    "Car Model",
                    available_models,
                    key=f"wizard_model_{brand}"
                )
            else:
                car_model = None
                st.warning("No models available for this manufacturer.")

            year = st.number_input(
                "Manufacturing Year",
                min_value=1990,
                max_value=2024,
                value=2018,
                step=1,
                key="wizard_year"
            )

            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                value=45000,
                step=1000,
                key="wizard_km"
            )

        with col2:
            st.markdown("#### Vehicle Specifications")

            fuel = st.selectbox(
                "Fuel Type",
                fuel_types,
                key="wizard_fuel"
            )

            transmission = st.selectbox(
                "Transmission",
                transmissions,
                key="wizard_transmission"
            )

            owner = st.selectbox(
                "Previous Owners",
                owners,
                key="wizard_owner"
            )

        if st.button(
            "Continue to Exterior Inspection →",
            use_container_width=True,
            disabled=not car_model
        ):

            car_details = {
                "Brand": brand,
                "model": car_model,
                "Year": int(year),
                "kmDriven": int(km_driven),
                "Transmission": transmission,
                "Owner": owner,
                "FuelType": fuel
            }

            try:
                with st.spinner("Estimating the base vehicle price..."):
                    base_price = predict_car_price(car_details)

                st.session_state.car_details = car_details
                st.session_state.base_price = base_price

                # Reset previous inspection data for this new valuation
                st.session_state.damage_results = []
                st.session_state.damage_analysis_done = False
                st.session_state.exterior_repair_cost_input = 0.0
                st.session_state.condition_ratings = {}

                st.session_state.workflow_step = 2
                st.session_state.scroll_to_top = True
                st.rerun()

            except Exception as e:
                st.error(f"Price prediction error: {e}")


# =================================================
# STEP 2: EXTERIOR DAMAGE INSPECTION
# =================================================

elif step == 2:

    st.markdown("### 🔍 Step 2: Exterior Condition")

    st.info(
        f"Your base vehicle estimate is ₹{st.session_state.base_price:,.0f}. "
        "Now let's inspect its exterior condition."
    )

    inspection_choice = st.radio(
        "How would you like to proceed?",
        [
            "Upload photos for damage inspection",
            "No visible damage — car is in good condition"
        ],
        key="inspection_choice"
    )

    if inspection_choice == "No visible damage — car is in good condition":

        st.session_state.damage_results = []
        st.session_state.damage_analysis_done = True
        st.session_state.exterior_repair_cost_input = 0.0

        st.success("No exterior damage has been reported.")

    else:

        uploaded_images = st.file_uploader(
            "Upload clear photographs of your vehicle",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="wizard_damage_upload"
        )

        st.caption(
            "Upload images showing scratches, dents, broken lights "
            "or other visible exterior damage."
        )

        if uploaded_images:

            if st.button(
                "Analyze Vehicle Damage",
                use_container_width=True
            ):

                all_results = []
                progress = st.progress(0)

                for index, uploaded_file in enumerate(uploaded_images):

                    try:
                        image = Image.open(uploaded_file).convert("RGB")

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
                            f"Could not process {uploaded_file.name}: {e}"
                        )

                    progress.progress(
                        (index + 1) / len(uploaded_images)
                    )

                st.session_state.damage_results = all_results
                st.session_state.damage_analysis_done = bool(all_results)

                combined_counts = {}

                for result in all_results:
                    for damage, count in result["damage_counts"].items():
                        combined_counts[damage] = (
                            combined_counts.get(damage, 0) + count
                        )

                suggested_cost = estimate_exterior_repair_cost(
                    combined_counts
                )

                st.session_state.exterior_repair_cost_input = suggested_cost

                if all_results:
                    st.success("Exterior inspection completed!")

        if st.session_state.damage_results:

            st.markdown("#### Inspection Results")

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

                    st.write("Detected damage:", result["damage_counts"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.workflow_step = 1
            st.session_state.scroll_to_top = True
            st.rerun()

    with col2:
        if st.button(
            "Continue to Mechanical Condition →",
            use_container_width=True,
            disabled=not st.session_state.damage_analysis_done
        ):
            st.session_state.workflow_step = 3
            st.session_state.scroll_to_top = True
            st.rerun()


# =================================================
# STEP 3: MECHANICAL CONDITION
# =================================================

elif step == 3:

    st.markdown("### ⚙️ Step 3: Mechanical Condition")

    st.write(
        "Select the condition of each component based on your "
        "inspection or the information provided by the vehicle owner."
    )

    st.warning(
        "These are user-reported conditions. CarWise AI does not "
        "mechanically diagnose the engine, gearbox, brakes or tyres."
    )

    st.markdown("#### Exterior Repair Estimate")

    damage_counts = {}

    for result in st.session_state.damage_results:
        for damage, count in result["damage_counts"].items():
            damage_counts[damage] = (
                damage_counts.get(damage, 0) + count
            )

    if damage_counts:
        st.write("Detected damage:", damage_counts)
    else:
        st.success("No visible exterior damage reported.")

    exterior_cost = st.number_input(
        "Estimated exterior repair cost (₹)",
        min_value=0.0,
        step=500.0,
        key="exterior_repair_cost_input"
    )

    st.caption(
        "This is a suggested estimate based on detected damage. "
        "Edit it to reflect actual repair quotations."
    )

    st.divider()

    st.markdown("#### Mechanical Component Assessment")

    condition_options = ["Good", "Average", "Poor"]

    condition_ratings = {}

    for component in ["Engine", "Transmission", "Brakes", "Tyres"]:

        condition_ratings[component] = st.selectbox(
            f"{component} condition",
            condition_options,
            key=f"condition_{component}"
        )

    mechanical_deduction, mechanical_breakdown = (
        calculate_mechanical_deduction(
            st.session_state.base_price,
            condition_ratings
        )
    )

    st.markdown("#### Estimated Mechanical Adjustments")

    for component, details in mechanical_breakdown.items():

        st.write(
            f"**{component}:** {details['condition']} — "
            f"{details['percentage']:.1f}% adjustment "
            f"(₹{details['deduction']:,.0f})"
        )

    st.metric(
        "Total estimated mechanical adjustment",
        f"₹{mechanical_deduction:,.0f}"
    )

    st.session_state.condition_ratings = condition_ratings

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.workflow_step = 2
            st.session_state.scroll_to_top = True
            st.rerun()

    with col2:
        if st.button(
            "Generate Final Valuation →",
            use_container_width=True
        ):
            st.session_state.workflow_step = 4
            st.session_state.scroll_to_top = True
            st.rerun()


# =================================================
# STEP 4: FINAL VALUATION
# =================================================

elif step == 4:

    st.markdown("### 📊 Step 4: Final Vehicle Valuation")

    base_price = float(st.session_state.base_price)

    exterior_cost = float(
        st.session_state.exterior_repair_cost_input
    )

    ratings = st.session_state.condition_ratings

    damage_counts = {}

    for result in st.session_state.damage_results:
        for damage, count in result["damage_counts"].items():
            damage_counts[damage] = (
                damage_counts.get(damage, 0) + count
            )

    mechanical_deduction, mechanical_breakdown = (
        calculate_mechanical_deduction(base_price, ratings)
    )

    total_deductions = exterior_cost + mechanical_deduction

    final_price = max(0, base_price - total_deductions)

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">
                FINAL INDICATIVE VEHICLE VALUATION
            </div>
            <div class="price-value">
                ₹{final_price:,.0f}
            </div>
            <div class="price-note">
                Calculated using the base AI estimate and condition adjustments.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Price Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Base AI Price", f"₹{base_price:,.0f}")

    with col2:
        st.metric("Exterior Repair Estimate", f"₹{exterior_cost:,.0f}")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Mechanical Adjustment",
            f"₹{mechanical_deduction:,.0f}"
        )

    with col4:
        st.metric(
            "Total Deductions",
            f"₹{total_deductions:,.0f}"
        )

    st.divider()

    st.markdown("#### Vehicle Condition Report")

    st.write("**Vehicle:**",
             f"{st.session_state.car_details['Brand']} "
             f"{st.session_state.car_details['model']}")

    st.write("**Manufacturing year:**",
             st.session_state.car_details["Year"])

    st.write("**Kilometres driven:**",
             f"{st.session_state.car_details['kmDriven']:,} km")

    st.markdown("#### Exterior Inspection")

    if damage_counts:
        st.json(damage_counts)
    else:
        st.success("No visible exterior damage reported.")

    st.markdown("#### Mechanical Assessment")

    for component, details in mechanical_breakdown.items():

        st.write(
            f"**{component}:** {details['condition']} "
            f"— estimated adjustment ₹{details['deduction']:,.0f}"
        )

    st.divider()

    st.caption(
        "This valuation is an indicative estimate based on historical "
        "vehicle listing data, visual damage detections and user-reported "
        "mechanical condition. It is not a professional inspection, "
        "repair quotation or guaranteed market price."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← Edit Condition Details", use_container_width=True):
            st.session_state.workflow_step = 3
            st.session_state.scroll_to_top = True
            st.rerun()

    with col2:
        if st.button("Start New Valuation", use_container_width=True):
            st.session_state.workflow_step = 1
            st.session_state.base_price = None
            st.session_state.car_details = None
            st.session_state.damage_results = []
            st.session_state.damage_analysis_done = False
            st.session_state.exterior_repair_cost_input = 0.0
            st.session_state.condition_ratings = {}
            st.rerun()



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
