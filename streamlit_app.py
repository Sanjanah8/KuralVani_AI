import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import base64

# Paths
MODEL_PATH = "models/tamil_slang_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
LOGO_PATH = "assets/tamillogo.png"

# Load model and encoder
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Feature extraction
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# UI Styling
st.set_page_config(page_title="தமிழ் வாசல் - Tamil Dialect Detector", layout="centered")

st.markdown("""
    <style>
    .main {
        font-family: 'Noto Sans Tamil', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 36px;
        color: #800000;
        font-weight: bold;
    }
    .sub {
        font-size: 20px;
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    .btn-upload {
        background-color: #c0392b;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Logo
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=120)

# Title
st.markdown('<div class="title">தமிழ் வட்டார வழக்கு கண்டறிதல்</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a Tamil audio file (.wav or .mp3) and detect its dialect</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("🎤 ஒலி கோப்பை பதிவேற்று", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp_audio.wav")
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        label = label_encoder.inverse_transform(predicted_class)[0]
        st.success(f"🔊 **கண்டறியப்பட்ட வட்டாரம்:** {label}")

        # Download option
        result_text = f"Prediction: {label}"
        b64 = base64.b64encode(result_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="prediction.txt">📥 Download Result</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"பிழை ஏற்பட்டது: {e}")
