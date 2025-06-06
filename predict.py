import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load once when module imported
model = tf.keras.models.load_model("models/tamil_slang_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def extract_mfcc(file):
    signal, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 40:
        mfcc = np.pad(mfcc, ((0,0), (0, 40 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :40]
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc.reshape(1, 40, 40, 1).astype(np.float32)
    return mfcc

def predict_dialect(audio_file):
    mfcc_input = extract_mfcc(audio_file)
    predictions = model.predict(mfcc_input)
    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = predictions[0][predicted_index]
    return predicted_label, confidence
