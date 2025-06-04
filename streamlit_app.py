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

# Extract MFCC features with fixed shape (40, 40, 1)
def extract_features(audio_file, max_pad_len=40):
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, 40, 40, 1)
    return mfcc

# Streamlit UI setup
st.set_page_config(page_title="தமிழ் வட்டார வழக்கு கண்டறிதல்", layout="centered")
st.title("📢 Tamil Dialect Classifier (தமிழ் வட்டார வழக்கு)")
st.write("Upload a Tamil audio file (WAV/MP3) to predict its regional dialect.")

audio_file = st.file_uploader("🎤 Upload Audio", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file)

    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    try:
        features = extract_features("temp_audio.wav")
        prediction = model.predict(features)

        if isinstance(prediction, list):
            prediction = prediction[0]

        predicted_class = np.argmax(prediction, axis=1)
        label = label_encoder.inverse_transform(predicted_class)
        st.success(f"🗣️ Predicted Dialect: **{label[0]}**")

    except Exception as e:
        # Fallback output on any error
        st.warning("⚠️ Could not predict accurately, showing fallback output.")
        st.success("🗣️ Predicted Dialect: **Unknown**")

    finally:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
