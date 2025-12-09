import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# LOAD MODEL & LABELS
model = load_model("keras_model.h5", compile=False)
labels = ["A", "B", "C"]

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# STREAMLIT UI
st.set_page_config(page_title="Real-Time Sign Detection", layout="centered")
st.title(" Real-Time Sign Language Detection")
st.markdown("Live camera-based sign recognition using Deep Learning")

prediction_box = st.empty()

# REAL-TIME VIDEO PROCESSOR
class SignProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
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

                imgModelInput = cv2.resize(imgWhite, (64, 64))
                imgModelInput = imgModelInput / 255.0
                imgModelInput = np.expand_dims(imgModelInput, axis=0)

                prediction = model.predict(imgModelInput)
                index = np.argmax(prediction)
                label = labels[index]

                prediction_box.success(f"Detected Sign: {label}")

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 120, y - offset), (255, 0, 255), cv2.FILLED)

                cv2.putText(imgOutput, label, (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)

                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            except:
                pass

        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

#START REAL-TIME CAMERA STREAM
webrtc_streamer(
    key="sign-realtime",
    video_processor_factory=SignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

