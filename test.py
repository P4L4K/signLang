import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# === Load model and labels ===
model = load_model("keras_model.h5")
labels = ["A", "B", "C", "Hello", "No", "Yes"]

# === Streamlit App ===
st.set_page_config(page_title="Sign Language Recognition", layout="centered")
st.title("🤟 Real-Time Sign Language Detection")
st.markdown("Using webcam + CNN model + cvzone HandTrackingModule")

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

cap = None

if run:
    cap = cv2.VideoCapture(0)
else:
    st.stop()

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to read from webcam.")
        break

    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Prepare image for prediction
            imgModelInput = cv2.resize(imgWhite, (64, 64))
            imgModelInput = imgModelInput / 255.0
            imgModelInput = np.expand_dims(imgModelInput, axis=0)

            prediction = model.predict(imgModelInput)
            index = np.argmax(prediction)
            label = labels[index]

            # Draw label
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except Exception as e:
            print("Crop or resize failed:", e)

    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(imgRGB)

cap.release()
cv2.destroyAllWindows()
