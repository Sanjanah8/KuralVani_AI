import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle

# Page config with favicon and title
st.set_page_config(
    page_title="குரல்வாணி AI - Tamil Dialect Detection",
    page_icon="assets/tamil-logo.jpeg",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load model and label encoder
model = tf.keras.models.load_model('models/tamil_slang_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# CSS Styling for buttons and general UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0B3954;
        margin-bottom: 0.1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #087E8B;
        margin-top: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .upload-text {
        font-size: 1rem;
        color: #555;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #0B3954;
        color: white;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 8px rgba(11, 57, 84, 0.3);
        margin-top: 15px;
    }
    .stButton>button:hover {
        background-color: #087E8B;
        cursor: pointer;
        box-shadow: 0 6px 12px rgba(8, 126, 139, 0.5);
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    .prediction-box {
        background-color: #E3F6F5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    img.waveform {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 320px;
        max-width: 90%;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(11,57,84,0.3);
        transition: transform 0.3s ease, filter 0.3s ease;
    }
    img.waveform:hover {
        filter: drop-shadow(0 0 12px #087E8B);
        transform: scale(1.1);
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Title & subtitle
st.markdown('<h1 class="main-header">குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">KuralVani AI - Tamil Dialect Detection</p>', unsafe_allow_html=True)

st.markdown("<p>தமிழ் பேசும் இடம் எங்கே என்று இந்த ஆடியோவால் கண்டறியுங்கள்.<br>Detect the regional Tamil dialect from this audio.</p>", unsafe_allow_html=True)

st.markdown('<p class="upload-text">தயவு செய்து கீழே உங்கள் தமிழ் வட்டார வழக்கு கொண்ட ஆடியோவை பதிவேற்றவும்.<br>Please upload your Tamil dialect audio file below.</p>', unsafe_allow_html=True)

# Waveform GIF
st.markdown(
    """
    <img class="waveform" src="assets/waveform.gif" alt="Waveform Animation" />
    """,
    unsafe_allow_html=True
)

# Audio uploader
audio_file = st.file_uploader("ஆடியோ பதிவேற்று (.wav, .mp3) / Upload audio (.wav, .mp3)", type=['wav', 'mp3'])

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Pad or truncate to ensure fixed length
    if mfccs.shape[1] < 40:
        pad_width = 40 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :40]
    return mfccs.T.reshape(1, 40, 40, 1).astype(np.float32)

if audio_file is not None:
    mfcc_features = extract_mfcc(audio_file)
    prediction = model.predict(mfcc_features)
    pred_label = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = float(np.max(prediction)) * 100

    # Display results
    st.markdown(f"""
    <div class="prediction-box">
        <h2>வட்டார வழக்கு: <span style="color:#0B3954;">{pred_label}</span> / Dialect: <span style="color:#0B3954;">{pred_label}</span></h2>
        <h3>நம்பிக்கை / Confidence: <span style="color:#087E8B;">{confidence:.2f}%</span></h3>
        <p>குறிப்பு: நமது மாதிரி 5 வட்டார வழக்குகளை முன்னிறுத்துகிறது: சென்னை, மதுரை, திருநெல்வேலி, இலங்கை மற்றும் ஸ்டாண்டர்ட் தமிழ்.</p>
        <p>Note: Our model predicts from 5 dialects: Chennai, Madurai, Tirunelveli, Sri Lanka, and Standard Tamil.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("உங்கள் தமிழ் வட்டார வழக்கு ஆடியோவை பதிவேற்றவும் / Please upload your Tamil dialect audio file.")

