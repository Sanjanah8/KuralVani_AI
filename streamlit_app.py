import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# ----------- Load your model and label encoder -----------

@st.cache_resource(show_spinner=True)
def load_model():
    model = tf.keras.models.load_model('models/tamil_slang_model.h5')
    return model

@st.cache_resource(show_spinner=True)
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return le

model = load_model()
label_encoder = load_label_encoder()

# ----------- App UI -----------

st.title("KuralVani AI - Tamil Slang Dialect Detection")

st.write("""
Upload a Tamil audio file (.wav) and the app will classify it into one of five dialects using a deep learning model.
""")

audio_file = st.file_uploader("Upload Audio", type=["wav"])

# ----------- Helper: extract features (MFCC) -----------

import librosa

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)  # Shape (1, 40)

# ----------- Predict button -----------

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Predict Dialect"):
        with st.spinner("Extracting features and predicting..."):
            # Extract MFCC features from uploaded file
            mfcc_features = extract_mfcc(audio_file)
            
            # Predict with the model
            preds = model.predict(mfcc_features)
            pred_class_index = np.argmax(preds, axis=1)[0]
            pred_class_label = label_encoder.inverse_transform([pred_class_index])[0]
            
            st.success(f"Predicted Dialect: **{pred_class_label}**")
            st.write(f"Confidence: {preds[0][pred_class_index]*100:.2f}%")
