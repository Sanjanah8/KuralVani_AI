import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

# Set page config with your app icon
st.set_page_config(
    page_title="குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்",
    page_icon="assets/tamillogo.jpeg",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load your trained model
model = tf.keras.models.load_model('models/tamil_slang_model.h5')

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Function to extract MFCC features from audio file
def extract_mfcc(audio_file, n_mfcc=40, max_len=174):
    # Load audio file with librosa
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Pad/truncate to fixed length for model input
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    # Model expects input shape (1, n_mfcc, max_len, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc

# CSS styling for pretty UI
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #FF6F61;
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF856F;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(255,111,97,0.4);
    }
    .stFileUploader>div {
        border: 2px dashed #FF6F61;
        border-radius: 12px;
        padding: 20px;
        background-color: #1E1E1E;
        color: #FFC1B3;
    }
    .confidence {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFD54F;
    }
    </style>
    """, unsafe_allow_html=True
)

# Show app logo
logo_img = Image.open("assets/tamillogo.jpeg")
st.image(logo_img, width=160, caption="குரல்வாணி AI - Tamil Dialect Detector")

# App title and description
st.title("குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்")
st.markdown(
    """
    தமிழில் பேசும் இடம் எங்கே என்று இந்த ஆடியோவால் கண்டறியுங்கள்.  
    Detect the regional Tamil dialect from this audio.
    """
)

# Upload audio file
audio_file = st.file_uploader("ஆடியோ பதிவேற்று (.wav, .mp3) / Upload audio (.wav, .mp3)", type=['wav','mp3'])

# Show waveform gif below uploader as animation
st.markdown("""
    <div style="text-align:center; margin: 10px 0;">
      <img src="https://media.giphy.com/media/l0MYA6PQFO1SxPaxu/giphy.gif" alt="Waveform" width="300" />
    </div>
    """, unsafe_allow_html=True)

if audio_file is not None:
    # Extract features and predict
    mfcc_features = extract_mfcc(audio_file)
    prediction = model.predict(mfcc_features)
    pred_index = np.argmax(prediction)
    pred_label = le.inverse_transform([pred_index])[0]
    confidence = prediction[0][pred_index] * 100

    # Show result with some spacing and styling
    st.markdown(f"### வட்டார வழக்கு / Dialect: **{pred_label}**")
    st.markdown(f"<p class='confidence'>நம்பிக்கை / Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

    # Extra info - you can customize this mapping if you have
    dialect_info = {
        "திருநெல்வேலி": "திருநெல்வேலி மாவட்டத்தின் ஆங்கிலத்தன்மை கொண்ட தமிழ் வட்டார வழக்கு.",
        "சென்னை": "சென்னை நகரத்தின் நகர்ப்புற தமிழ் வட்டார வழக்கு.",
        "கோயம்புத்தூர்": "கோயம்புத்தூர் மாவட்டத்தின் ஆங்கிலத்தன்மை கொண்ட தமிழ் வட்டார வழக்கு.",
        "மதுரை": "மதுரை மாவட்டத்தின் ஆங்கிலத்தன்மை கொண்ட தமிழ் வட்டார வழக்கு.",
        "இலங்கை தமிழ்": "இலங்கையில் பேசப்படும் தமிழ் வட்டார வழக்கு.",
        # Add more as per your label encoder classes
    }

    extra_text = dialect_info.get(pred_label, "இந்த வட்டார வழக்கு பற்றிய கூடுதல் தகவல் கிடைக்கவில்லை.")
    st.markdown(f"**குறிப்பு / Note:** {extra_text}")

else:
    st.info("தயவு செய்து தமிழ் வட்டார வழக்கு கொண்ட ஆடியோவை பதிவேற்றவும். / Please upload an audio file with Tamil dialect.")

# Footer
st.markdown(
    """
    <hr style="border:1px solid #FF6F61;">
    <p style="text-align:center; color:#B0B0B0;">
    © 2025 குரல்வாணி AI - Tamil Dialect Detection
    </p>
    """,
    unsafe_allow_html=True,
)
