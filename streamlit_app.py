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
    page_icon="assets/tamil-logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Background CSS (optional) ---
st.markdown(
    """
    <style>
    body {
        background-color: #f9f6f2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6b2c2c;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #a34444;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #a34444;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: background-color 0.3s ease;
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
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Logo ---
logo = Image.open('assets/tamil-logo.png')
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
    # Load audio with librosa
    signal, sr = librosa.load(file, sr=22050)
    # Extract MFCCs with 40 coefficients, fixed length 40 frames
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=40)
    # Pad or truncate to 40 frames
    if mfcc.shape[1] < 40:
        pad_width = 40 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :40]
    # Normalize mfcc
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    # Reshape for model: (40, 40, 1)
    mfcc = mfcc[..., np.newaxis]
    return mfcc

if audio_file is not None:
    with st.spinner("பதிலுக்கு கணினி கணக்கிடுகிறது... / Computing prediction..."):
        try:
            mfccs = extract_mfcc(audio_file)
            mfccs = np.expand_dims(mfccs, axis=0)  # batch dimension

            prediction = model.predict(mfccs)
            pred_index = np.argmax(prediction)
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            confidence = prediction[0][pred_index] * 100

            dialect_map = {
                'chennai': 'சென்னை (Chennai)',
                'madurai': 'மதுரை (Madurai)',
                'tirunelveli': 'திருநெல்வேலி (Tirunelveli)',
                'srilanka': 'இலங்கை (Sri Lanka)',
                'standard': 'ஸ்டான்டர்டு தமிழ் (Standard Tamil)'
            }

            pred_label_tamil = dialect_map.get(pred_label.lower(), pred_label)

            st.success(f"வட்டார வழக்கு: {pred_label_tamil} / Dialect: {pred_label_tamil}")
            st.markdown(f'<p class="confidence">நம்பிக்கை / Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ பிழை ஏற்பட்டது: {str(e)}")

# --- Footer ---
st.markdown("---")
st.write("**Developed by KuralVani AI Team** | © 2025")


