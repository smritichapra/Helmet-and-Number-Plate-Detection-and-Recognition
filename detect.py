import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import imutils
from keras.models import load_model

# ------------------------------------------------------------
# ✅ Ensure the temp folder exists before saving uploaded videos
# ------------------------------------------------------------
if not os.path.exists("temp"):
    os.makedirs("temp")

# ------------------------------------------------------------
# Load YOLO model
# Load helmet detection model
# GitHub release URL (replace with your own link)
url = "https://github.com/smritichapra/Helmet-and-Number-Plate-Detection-and-Recognition/releases/tag/hi"

# Download the model file
urllib.request.urlretrieve(url, "yolov3-custom_7000.weights")
# ------------------------------------------------------------
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")

# Try using GPU if available, else fallback to CPU
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

except:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    st.write("⚙️ CUDA not found, using CPU mode")

# ------------------------------------------------------------
# Load helmet detection model
# GitHub release URL (replace with your own link)
url = "https://github.com/smritichapra/Helmet-and-Number-Plate-Detection-and-Recognition/releases/tag/hii"

# Download the model file
urllib.request.urlretrieve(url, "helmet-nonhelmet_cnn.h5")

# ------------------------------------------------------------
model = load_model('helmet-nonhelmet_cnn.h5')

st.title("Bike, Helmet, and Number Plate Detection and Recognition")

# ------------------------------------------------------------
# Upload and process video
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Choose a video file", type=["mp4", "avi"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = os.path.join("temp", uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    video = cv2.VideoCapture(temp_file_path)

    if not video.isOpened():
        st.error("❌ Error: Could not open video file.")
    else:
        stframe = st.empty()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame = imutils.resize(frame, height=500)
            height, width = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = [net.forward("yolo_82"), net.forward("yolo_94"), net.forward("yolo_106")]

            confidences, boxes, classIds = [], [], []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    if classIds[i] == 0:  # bike
                        helmet_roi = frame[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                        if helmet_roi.shape[0] > 0 and helmet_roi.shape[1] > 0:
                            helmet_roi = cv2.resize(helmet_roi, (224, 224))
                            helmet_roi = np.array(helmet_roi, dtype='float32') / 255.0
                            helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
                            prediction = int(model.predict(helmet_roi)[0][0])
                            label = "Helmet" if prediction == 0 else "No Helmet"
                            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                            cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)


        video.release()
        # Cleanup temporary video
        os.remove(temp_file_path)
        st.success("✅ Processing complete! Temporary file removed.")

