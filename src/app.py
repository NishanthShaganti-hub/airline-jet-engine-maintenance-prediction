import streamlit as st
import joblib
import pandas as pd
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# AVIATION THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        180deg,
        #020617 0%,
        #0F172A 50%,
        #1E293B 100%
    );
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0B1220;
    border-right: 1px solid #1E40AF;
}

/* Main Title */
h1 {
    color: #E2E8F0 !important;
    font-weight: 800 !important;
}

/* Section Headers */
h2, h3 {
    color: #7DD3FC !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1E40AF;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 0px 12px rgba(59,130,246,0.15);
}

/* Info Box */
.stInfo {
    background-color: #0F2744;
    border-left: 5px solid #38BDF8;
}

/* Success Box */
.stSuccess {
    background-color: rgba(34,197,94,0.15);
}

/* Warning Box */
.stWarning {
    background-color: rgba(245,158,11,0.15);
}

/* Error Box */
.stError {
    background-color: rgba(239,68,68,0.15);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    color: #CBD5E1;
    font-size: 16px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
}

/* Buttons */
.stButton > button {
    background-color: #0284C7;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #0369A1;
    color: white;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    "models/xgb_rul_model.pkl"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------



st.title("✈️ Aircraft Engine Predictive Maintenance Dashboard")

st.caption(
    "NASA CMAPSS • XGBoost • SHAP Explainability • Predictive Maintenance"
)

st.markdown(
    """
    ### AI-Powered Aircraft Engine Health Monitoring

    Predict Remaining Useful Life (RUL) using NASA CMAPSS
    sensor data, XGBoost Regression and SHAP Explainability.
    """
)



# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("🛫 Flight Sensor Inputs")

cycle = st.sidebar.number_input(
    "Cycle",
    min_value=0,
    value=100
)

sensor_2 = st.sidebar.number_input("Sensor 2")
sensor_3 = st.sidebar.number_input("Sensor 3")
sensor_4 = st.sidebar.number_input("Sensor 4")
sensor_7 = st.sidebar.number_input("Sensor 7")
sensor_11 = st.sidebar.number_input("Sensor 11")
sensor_12 = st.sidebar.number_input("Sensor 12")
sensor_15 = st.sidebar.number_input("Sensor 15")
sensor_17 = st.sidebar.number_input("Sensor 17")
sensor_20 = st.sidebar.number_input("Sensor 20")
sensor_21 = st.sidebar.number_input("Sensor 21")

predict_btn = st.sidebar.button("🚀 Predict RUL")





# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------
prediction = None

if predict_btn:

    input_data = pd.DataFrame(
        [[
            sensor_2,
            sensor_3,
            sensor_4,
            sensor_7,
            sensor_11,
            sensor_12,
            sensor_15,
            sensor_17,
            sensor_20,
            sensor_21,
            cycle
        ]],
        columns=[
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_11",
            "sensor_12",
            "sensor_15",
            "sensor_17",
            "sensor_20",
            "sensor_21",
            "cycle"
        ]
    )

    prediction = model.predict(input_data)[0]

st.header("🛬 Engine Health Assessment")

if prediction is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted RUL",
            f"{prediction:.2f} Cycles"
        )

    with col2:

        if prediction > 50:
            st.success(
                "🟢 SAFE — Engine operating normally"
            )

        elif prediction > 20:
            st.warning(
                "🟡 WARNING — Maintenance should be scheduled soon"
            )

        else:
            st.error(
                "🔴 CRITICAL — Immediate maintenance recommended"
            )

else:

    st.info(
        "Enter engine sensor values and click 'Predict RUL' to generate a prediction."
    )




# --------------------------------------------------
# VISUALIZATION SECTION
# --------------------------------------------------

st.header("✈️ Aircraft Analytics Center")

tab1, tab2, tab3 = st.tabs(
    [
    "Feature Insights",
    "Explainability",
    "Model Learning"
]
)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("📡 Fleet Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        "26.68 Cycles"
    )

with col2:
    st.metric(
        "RMSE",
        "36.77 Cycles"
    )

with col3:
    st.metric(
        "R² Score",
        "70.4%"
)

st.divider()


# ---------------------------
# Feature Importance
# ---------------------------

with tab1:

    st.subheader("Feature Importance")

    if os.path.exists(
        "results/feature_importance.png"
    ):
        st.image(
            "results/feature_importance.png",
            use_container_width=True
        )
    else:
        st.warning(
            "feature_importance.png not found"
        )

# ---------------------------
# SHAP
# ---------------------------

with tab2:

    st.subheader(
        "SHAP Explainability"
    )

    if os.path.exists(
        "results/shap_summary.png"
    ):
        st.image(
            "results/shap_summary.png",
            use_container_width=True
        )
    else:
        st.warning(
            "shap_summary.png not found"
        )

# ---------------------------
# Learning Curve
# ---------------------------

with tab3:

    st.subheader(
        "Learning Curve"
    )

    if os.path.exists(
        "results/learning_curve.png"
    ):
        st.image(
            "results/learning_curve.png",
            use_container_width=True
        )
    else:
        st.warning(
            "learning_curve.png not found"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "✈ Built using Python • XGBoost • SHAP • Streamlit • NASA CMAPSS Dataset"
)
st.markdown("---")

st.caption(
    "Developed by S.Nishanth | Aircraft Predictive Maintenance System"
)