
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import tempfile

# Load the trained model and label encoder
model = tf.keras.models.load_model("models/tamil_slang_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Extract MFCC features from audio file
def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    # Ensure shape is (40, 40)
    if mfcc.shape[1] < 40:
        mfcc = np.pad(mfcc, ((0, 0), (0, 40 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :40]

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Reshape for CNN input: (1, 40, 40, 1)
    mfcc = mfcc.reshape(1, 40, 40, 1).astype(np.float32)

    return mfcc

# Predict dialect
def predict_dialect(file_path):
    features = extract_mfcc(file_path)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label

# Streamlit UI
st.set_page_config(page_title="KuralVani AI", layout="centered")
st.title("🗣️ KuralVani AI")
st.markdown("### தமிழ் வட்டார வழக்கு கண்டறிதல்")
st.write("குரல் கோப்பை வழங்கவும் (.wav) – மொழிப் பேச்சை அடிப்படையாகக் கொண்டு வட்டார வழக்கைக் கண்டறிகிறது.")

# File uploader
uploaded_file = st.file_uploader("🎤 உங்கள் குரல் (.wav) கோப்பை பதிவேற்றவும்:", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔍 கணிப்பிக்கிறது..."):
        try:
            prediction = predict_dialect(tmp_path)
            st.success(f"✅ கண்டறியப்பட்ட வட்டார வழக்கு: **{prediction}**")
        except Exception as e:
            st.error(f"⚠️ பிழை ஏற்பட்டது: {str(e)}")

