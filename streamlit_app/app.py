
import streamlit as st
import requests
from PIL import Image
import io

# FastAPI server URL (replace with the actual URL if running locally)
SERVER_URL = "http://localhost:8000/analyze_emotion"

st.title("Emotion Detection from Images")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert image to byte array
    image_bytes = uploaded_file.read()
    response = requests.post(SERVER_URL, files={"file": image_bytes})

    if response.status_code == 200:
        emotion = response.json().get("emotion")
        st.write(f"Detected Emotion: {emotion}")
    else:
        st.write("Error occurred during image classification.")
