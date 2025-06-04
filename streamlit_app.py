import streamlit as st
from predict import predict_dialect
from PIL import Image

# Dialect map for display
dialect_map = {
    'chennai': 'சென்னை (Chennai)',
    'madurai': 'மதுரை (Madurai)',
    'thirunelveli': 'திருநெல்வேலி (Tirunelveli)',
    'srilanka': 'இலங்கை (Sri Lanka)',
    'no_slang': 'ஸ்டான்டர்டு தமிழ் (Standard Tamil)'
}

# Page config & styling (add your CSS as needed)
st.set_page_config(page_title="குரல்வாணி AI - Tamil Dialect Detection", page_icon="assets/tamillogo.jpeg")

logo = Image.open('assets/tamillogo.jpeg')
st.image(logo, width=120)

st.markdown('<h1 style="color:white; text-align:center;">குரல்வாணி AI - தமிழ் வட்டார வழக்கு கண்டறிதல்</h1>', unsafe_allow_html=True)
st.write("தமிழ் பேசும் இடத்தை கண்டறிய ஆடியோ பதிவேற்றவும்.")

audio_file = st.file_uploader("ஆடியோ (.wav, .mp3) பதிவேற்றவும்", type=["wav", "mp3"])

if audio_file is not None:
    with st.spinner("கணிப்பை கணக்கிடுகிறது..."):
        try:
            label, confidence = predict_dialect(audio_file)
            display_label = dialect_map.get(label.lower(), label)
            st.markdown(f"**வட்டார வழக்கு:** {display_label}")
            st.markdown(f"**நம்பிக்கை:** {confidence * 100:.2f}%")
        except Exception as e:
            st.error(f"பிழை ஏற்பட்டது: {str(e)}")
