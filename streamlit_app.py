import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # shape (40, time_steps)
    max_pad_len = 40
    
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    mfccs = mfccs[..., np.newaxis]  # shape (40, 40, 1)
    mfccs = np.expand_dims(mfccs, axis=0)  # shape (1, 40, 40, 1)
    return mfccs.astype(np.float32)

dialect_labels = [
    'சென்னை (Chennai)', 
    'மதுரை (Madurai)', 
    'திருநெல்வேலி (Tirunelveli)', 
    'இலங்கை (Sri Lanka)', 
    'ஸ்டாண்டர்ட் (Standard)'
]

st.title('குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்\nKuralVani AI - Tamil Dialect Detection')

st.write('''  
தமிழ் பேசும் இடம் எங்கே என்று இந்த ஆடியோவால் கண்டறியுங்கள்.  
Detect the regional Tamil dialect from this audio.  

தயவு செய்து கீழே உங்கள் தமிழ் வட்டார வழக்கு கொண்ட ஆடியோவை பதிவேற்றவும்.  
Please upload your Tamil dialect audio file below.  
''')

uploaded_file = st.file_uploader("ஆடியோ பதிவேற்று (.wav, .mp3) / Upload audio (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner('மாதிரி கணக்கிடுகிறது... தயவு செய்து காத்திருங்கள். / Predicting... please wait.'):
        mfcc_features = extract_mfcc(uploaded_file)
        preds = model.predict(mfcc_features)
        pred_index = np.argmax(preds)
        pred_label = dialect_labels[pred_index]
        confidence = preds[0][pred_index] * 100

    st.success(f"வட்டார வழக்கு: **{pred_label}** / Dialect: **{pred_label}**")
    st.write(f"நம்பிக்கை / Confidence: {confidence:.2f}%")
