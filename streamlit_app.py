import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

# --- Load model and label encoder ---
model = tf.keras.models.load_model('models/tamil_slang_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --- Page config ---
st.set_page_config(
    page_title="குரல்வாணி AI - Tamil Dialect Detection",
    page_icon="assets/tamillogo.jpeg",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CSS styling ---
st.markdown(
    """
    <style>
    .title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #6b2c2c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #a34444;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #a34444;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #7f2a2a;
        cursor: pointer;
    }
    .result {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #5a2b2b;
    }
    .confidence {
        text-align: center;
        font-size: 1.1rem;
        color: #4b403f;
    }
    .gif-container {
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Logo ---
st.image("assets/tamillogo.jpeg", width=120)

# --- Title and subtitle ---
st.markdown('<div class="title">குரல்வாணி AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">தமிழ் வட்டார வழக்கு கண்டறிதல் (Tamil Dialect Detection)</div>', unsafe_allow_html=True)

st.write("📢 **Upload your Tamil audio to detect the dialect region.**")
st.write("---")

# --- File upload ---
audio_file = st.file_uploader("Upload Audio File (.wav or .mp3)", type=['wav', 'mp3'])

# --- Feature extractor ---
def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 40:
        pad_width = 40 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :40]

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return np.expand_dims(mfcc.reshape(40, 40, 1), axis=0)  # (1, 40, 40, 1)

# --- Label mapping ---
dialect_map = {
    'chennai': 'சென்னை (Chennai)',
    'madurai': 'மதுரை (Madurai)',
    'thirunelveli': 'திருநெல்வேலி (Tirunelveli)',
    'srilanka': 'இலங்கை (Sri Lanka)',
    'no_slang': 'ஸ்டான்டர்டு தமிழ் (Standard Tamil)'
}

# --- Prediction block ---
if audio_file is not None:
    with st.spinner("🔎 Processing and predicting..."):
        st.markdown('<div class="gif-container"><img src="https://i.gifer.com/origin/5f/5f8ed9e745a502a54ff52e37496e715a.gif" width="100"/></div>', unsafe_allow_html=True)
        
        try:
            mfcc_input = extract_mfcc(audio_file)
            prediction = model.predict(mfcc_input)
            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index] * 100
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            tamil_label = dialect_map.get(pred_label.lower(), pred_label)

            st.markdown(f'<div class="result">🗣 வட்டார வழக்கு: {tamil_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">நம்பிக்கை: {confidence:.2f}%</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing audio: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("🔬 Developed by **KuralVani AI Team** | © 2025", unsafe_allow_html=True)
