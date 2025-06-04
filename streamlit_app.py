import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import librosa

# ----------- Load your model and label encoder -----------

@st.cache_resource(show_spinner=True)
def load_model():
    model = tf.keras.models.load_model('models/tamil_slang_model.h5')
    return model

@st.cache_resource(show_spinner=True)
def load_label_encoder():
    with open('models/label_encoder.pkl', 'rb') as f:
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

def extract_mfcc(file, max_pad_len=40):
    y, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Reshape to (1, 40, 40, 1) to match model input
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    return mfccs

# ----------- Predict button -----------

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Predict Dialect"):
        with st.spinner("Extracting features and predicting..."):
            try:
                mfcc_features = extract_mfcc(audio_file)
                preds = model.predict(mfcc_features)
                pred_class_index = np.argmax(preds, axis=1)[0]
                pred_class_label = label_encoder.inverse_transform([pred_class_index])[0]
                confidence = preds[0][pred_class_index] * 100

                st.success(f"Predicted Dialect: **{pred_class_label}**")
                st.write(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
