from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = FastAPI()

# Update this path to where the model is located on your system
model_path = "C:/Users/rishi/Downloads/emotion_detection_project/model/emotion_cnn_model.h5"
visual_model = load_model(model_path)


@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    # Convert image to format required by the model
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict emotion
    predictions = visual_model.predict(image)
    emotion_index = np.argmax(predictions[0])
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion = emotions[emotion_index]
    return {"emotion": emotion}
