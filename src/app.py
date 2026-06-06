import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# Load model
model = joblib.load(
    "models/xgb_rul_model.pkl"
)
st.title(
    "✈️ Aircraft Engine Predictive Maintenance System"
)

st.markdown(
    """
    Predict Remaining Useful Life (RUL) using
    machine learning and engine sensor data.
    """
)
cycle = st.number_input(
    "Cycle",
    min_value=0,
    value=100
)

sensor_2 = st.number_input("sensor_2")
sensor_3 = st.number_input("sensor_3")
sensor_4 = st.number_input("sensor_4")
sensor_7 = st.number_input("sensor_7")
sensor_11 = st.number_input("sensor_11")
sensor_12 = st.number_input("sensor_12")
sensor_15 = st.number_input("sensor_15")
sensor_17 = st.number_input("sensor_17")
sensor_20 = st.number_input("sensor_20")
sensor_21 = st.number_input("sensor_21")

if st.button("Predict RUL"):

    input_data = pd.DataFrame([[
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
    ])

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"Predicted RUL: {prediction:.2f} cycles"
    )

    if prediction > 50:
        st.success("SAFE")

    elif prediction > 20:
        st.warning("WARNING")

    else:
        st.error("CRITICAL")
        
# TO RUN THIS
# python -m streamlit run src/app.py