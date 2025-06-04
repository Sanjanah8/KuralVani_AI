# streamlit_app.py

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os

# Load the model and encoder
MODEL_PATH = "models/tamil_slang_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract MFCCs
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Streamlit UI
st.set_page_config(page_title="தமிழ் வட்டார வழக்கு கண்டறிதல்", layout="centered")

st.title("📢 Tamil Dialect Classifier (தமிழ் வட்டார வழக்கு)")
st.write("Upload a Tamil audio file (WAV/MP3) to predict its regional dialect.")

audio_file = st.file_uploader("🎤 Upload Audio", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file)

    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    try:
        features = extract_features("temp_audio.wav")
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        label = label_encoder.inverse_transform(predicted_class)

        st.success(f"🗣️ Predicted Dialect: **{label[0]}**")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
