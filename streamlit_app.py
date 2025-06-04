import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pickle
import base64

# Set page config with favicon (app logo)
st.set_page_config(
    page_title="குரல்வாணி AI - Tamil Dialect Detection",
    page_icon="assets/tamillogo.jpeg",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load model and label encoder
model = tf.keras.models.load_model('models/tamil_slang_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Background image style with low opacity
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_string}");
            background-size: cover;
            background-position: center;
            opacity: 0.07;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('assets/tamil-logo.png')

# Custom CSS for UI styling & animations
st.markdown("""
<style>
/* Main container padding */
.main-container {
    padding: 1.5rem 2rem;
    max-width: 700px;
    margin: auto;
    background-color: #fefefe;
    border-radius: 15px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    transition: all 0.3s ease;
}

/* Title styles */
h1, h2 {
    font-family: 'Noto Serif Tamil', serif;
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0.3rem;
}

h2 {
    font-weight: 500;
    color: #34495e;
}

/* Upload button style */
.stFileUpload button {
    background-color: #1abc9c;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 0 4px 12px rgba(26, 188, 156, 0.3);
}

.stFileUpload button:hover {
    background-color: #16a085;
    transform: scale(1.05);
    cursor: pointer;
}

/* Predict button */
#predict-btn {
    background-color: #2980b9;
    color: white;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    margin-top: 1rem;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 0 5px 15px rgba(41, 128, 185, 0.4);
}

#predict-btn:hover {
    background-color: #1f618d;
    transform: scale(1.1);
    cursor: pointer;
}

/* Confidence bar container */
.confidence-bar {
    background-color: #ecf0f1;
    border-radius: 20px;
    height: 25px;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

/* Confidence bar fill */
.confidence-fill {
    height: 100%;
    background-color: #27ae60;
    width: 0%;
    border-radius: 20px;
    transition: width 1s ease-in-out;
}

/* Result text */
.result-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 0.5rem;
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-top: 3rem;
    font-family: 'Noto Serif Tamil', serif;
}
</style>
""", unsafe_allow_html=True)

# Main container div
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Titles in Tamil and English
st.markdown("# குரல்வாணி AI")
st.markdown("## Tamil Dialect Detection (தமிழ் வட்டார வழக்கு கண்டறிதல்)")
st.markdown(
    """
    ### தமிழ் பேசும் இடம் எங்கே என்று இந்த ஆடியோவால் கண்டறியுங்கள்.<br>
    ### Detect the regional Tamil dialect from this audio.
    """,
    unsafe_allow_html=True,
)

# File uploader
audio_file = st.file_uploader(
    "ஆடியோ பதிவேற்று (.wav, .mp3) / Upload audio (.wav, .mp3)",
    type=['wav', 'mp3'],
    help="Upload your Tamil dialect audio file here."
)

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Pad or truncate to fixed length 40x40
    if mfccs.shape[1] < 40:
        pad_width = 40 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :40]
    return mfccs.T.reshape(1,40,40,1)

def predict_dialect(mfcc):
    preds = model.predict(mfcc)
    pred_index = np.argmax(preds)
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = preds[0][pred_index]
    return pred_label, confidence

if audio_file:
    st.audio(audio_file, format='audio/wav')
    mfccs = extract_mfcc(audio_file)
    pred_label, confidence = predict_dialect(mfccs)

    # Show prediction results with confidence bar
    st.markdown(f"### வட்டார வழக்கு / Dialect: **{pred_label}**")
    st.markdown(f"### நம்பிக்கை / Confidence: {confidence * 100:.2f}%")

    # Confidence bar HTML & animation
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Extra info in Tamil and English
    dialect_info = {
        "Chennai": "சென்னை வட்டார வழக்கு - சென்னை, தமிழ்நாடு",
        "Madurai": "மதுரை வட்டார வழக்கு - மதுரை, தமிழ்நாடு",
        "Tirunelveli": "திருநெல்வேலி வட்டார வழக்கு - தென் தமிழ்நாடு",
        "Sri Lanka": "இலங்கை தமிழ் வட்டார வழக்கு",
        "Standard": "முழுமையான தமிழ் மொழி (Standard Tamil)",
    }

    info_text = dialect_info.get(pred_label, "Unknown dialect information not available.")
    st.markdown(f"**வட்டார வழக்கு விளக்கம் / Dialect Info:** {info_text}")

    # Show raw prediction probabilities table
    st.markdown("#### மற்ற வாய்ப்புகள் / Other Probabilities:")
    pred_probs = model.predict(mfcc)[0]
    prob_dict = {label_encoder.inverse_transform([i])[0]: f"{prob*100:.2f}%" for i, prob in enumerate(pred_probs)}
    st.table(prob_dict)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 குரல்வாணி AI | KuralVani AI - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
