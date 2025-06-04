# streamlit_app.py

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load model and label encoder once (put your actual model paths here)
MODEL_PATH = "models/tamil_slang_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

@st.cache_resource  # Cache so model loads once
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

def extract_mfcc(file):
    signal, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 40:
        mfcc = np.pad(mfcc, ((0,0), (0, 40 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :40]

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc.reshape(1, 40, 40, 1).astype(np.float32)
    return mfcc

def predict_dialect(audio_file):
    mfcc_input = extract_mfcc(audio_file)
    # Debug print, comment out in production
    # st.write("MFCC input shape:", mfcc_input.shape)

    predictions = model.predict(mfcc_input)
    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = predictions[0][predicted_index]
    return predicted_label, confidence

# --- Streamlit UI starts here ---

st.title("Tamil Dialect / Slang Detection")
st.write("Upload a Tamil audio clip to detect the dialect/slang")

uploaded_file = st.file_uploader("Choose an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    try:
        # Predict on uploaded file
        label, confidence = predict_dialect(uploaded_file)
        st.success(f"Predicted Dialect: **{label}**")
        st.info(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
