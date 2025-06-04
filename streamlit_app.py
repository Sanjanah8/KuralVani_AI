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
    body {
        background-color: #f9f6f2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #6b2c2c;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #a34444;
        margin-top: 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stButton>button {
        background-color: #a34444;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: background-color 0.3s ease;
        display: block;
        margin: 0 auto;
    }
    .stButton>button:hover {
        background-color: #7f2a2a;
        cursor: pointer;
    }
    .confidence {
        font-weight: 700;
        color: #4b403f;
        font-size: 1.1rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #5a2b2b;
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
logo = Image.open('assets/tamillogo.jpeg')
st.image(logo, width=120)

# --- Title and instructions ---
st.markdown('<h1 class="title">குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">KuralVani AI - Tamil Dialect Detection</h3>', unsafe_allow_html=True)

st.write("**தமிழ் பேசும் இடம் எங்கே என்று இந்த ஆடியோவால் கண்டறியுங்கள்.**")
st.write("**Detect the regional Tamil dialect from this audio.**")
st.write("---")

# --- File uploader ---
audio_file = st.file_uploader(
    "தயவு செய்து கீழே உங்கள் தமிழ் வட்டார வழக்கு கொண்ட ஆடியோவை பதிவேற்றவும்.\nPlease upload your Tamil dialect audio file (.wav, .mp3)",
    type=['wav', 'mp3']
)

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Ensure (40, 40)
    if mfcc.shape[1] < 40:
        pad_width = 40 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :40]

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc.reshape(40, 40, 1)  # Shape (40, 40, 1)
    return np.expand_dims(mfcc, axis=0)  # Final shape (1, 40, 40, 1)

# Mapping English labels to Tamil
dialect_map = {
    'chennai': 'சென்னை (Chennai)',
    'madurai': 'மதுரை (Madurai)',
    'tirunelveli': 'திருநெல்வேலி (Tirunelveli)',
    'srilanka': 'இலங்கை (Sri Lanka)',
    'standard': 'ஸ்டான்டர்டு தமிழ் (Standard Tamil)'
}

# --- Main prediction block ---
if audio_file is not None:
    with st.spinner("பதிலுக்கு கணினி கணக்கிடுகிறது... / Computing prediction..."):
        st.markdown(
            '<div class="gif-container">'
            '<img src="https://i.gifer.com/origin/5f/5f8ed9e745a502a54ff52e37496e715a.gif" width="120" alt="waveform gif"/>'
            '</div>', 
            unsafe_allow_html=True
        )

        try:
            mfcc_input = extract_mfcc(audio_file)  # shape: (1, 40, 40, 1)
            prediction = model.predict(mfcc_input)

            pred_index = np.argmax(prediction)
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            confidence = prediction[0][pred_index] * 100

            pred_label_tamil = dialect_map.get(pred_label.lower(), pred_label)

            st.markdown(f'<p class="result">வட்டார வழக்கு: {pred_label_tamil} / Dialect: {pred_label_tamil}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">நம்பிக்கை / Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ பிழை ஏற்பட்டது: {str(e)}")

# --- Footer ---
st.markdown("---")
st.write("**Developed by KuralVani AI Team** | © 2025")
