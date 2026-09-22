import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Load model
model = tf.keras.models.load_model("machine_temperature_rnn.keras")


# Training data used for scaling
data = np.array([
    [60, 2.1],
    [62, 2.3],
    [61, 2.2],
    [64, 2.5],
    [65, 2.6],
    [67, 2.7],
    [66, 2.5],
    [69, 2.8],
    [70, 2.9],
    [72, 3.0],
    [71, 2.9],
    [74, 3.1],
    [75, 3.2],
    [77, 3.3],
    [76, 3.1],
    [79, 3.4],
    [80, 3.5],
    [82, 3.6],
    [81, 3.4],
    [83, 3.6]
])


# Create scaler
scaler = MinMaxScaler()
scaler.fit(data)


# Title
st.title("Machine Temperature Prediction using RNN")

st.write("Enter temperature and vibration values for the previous two timestamps.")


# T21
st.subheader("Previous Timestamp 1")

temp1 = st.number_input(
    "Temperature T21 (°C)",
    value=81.0
)

vibration1 = st.number_input(
    "Vibration T21",
    value=3.5
)


# T22
st.subheader("Previous Timestamp 2")

temp2 = st.number_input(
    "Temperature T22 (°C)",
    value=83.0
)

vibration2 = st.number_input(
    "Vibration T22",
    value=3.6
)


# Prediction button
if st.button("Predict Next Temperature"):

    new_data = np.array([
        [temp1, vibration1],
        [temp2, vibration2]
    ])

    # Scale input
    new_data_scaled = scaler.transform(new_data)

    # Reshape for RNN
    new_input = new_data_scaled.reshape(1, 2, 2)

    # Predict
    prediction_scaled = model.predict(
        new_input,
        verbose=0
    )

    # Convert prediction back to Celsius
    dummy = np.zeros((1, 2))
    dummy[0, 0] = prediction_scaled[0][0]

    prediction = scaler.inverse_transform(dummy)

    predicted_temperature = prediction[0][0]

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
