import cv2
import numpy as np

# 1. Initialize the Webcam and Face Detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/mitushijain/Desktop/Face_scanning code/haarcascade_frontalface_default.xml')

# 2. Initialize the Modern ONNX Networks
AGE_ONNX = "/Users/mitushijain/Desktop/Face_scanning code/age_googlenet.onnx"
# GENDER_ONNX = "/Users/mitushijain/Desktop/Face_scanning code/gender_googlenet.onnx"

age_net = cv2.dnn.readNetFromONNX(AGE_ONNX)
# gender_net = cv2.dnn.readNetFromONNX(GENDER_ONNX)

# Technical constants required by the GoogLeNet models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

faces_data = []
i = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Grab the clean face area before blurring it
        crop_img = frame[y:y + h, x:x + w, :]

        if crop_img.size == 0:
            continue

        # Your dataset collection logic (saves CLEAN unblurred faces)
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1

        # === DEEP LEARNING ANALYSIS LOGIC (AGE & GENDER) ===
        try:
            # Format the image into a deep learning blob (227x227 pixels)
            blob = cv2.dnn.blobFromImage(crop_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            predicted_age = AGE_BUCKETS[age_preds.argmax()]

            # Predict Gender
            # gender_net.setInput(blob)
            # gender_preds = gender_net.forward()
            # predicted_gender = GENDER_LIST[gender_preds.argmax()]

            # Combine them for the screen overlay
            label_text = f"{predicted_age}"
        except Exception as e:
            label_text = "Analysis Failed"

        # === FACE BLURRING LOGIC ===
        face_roi = frame[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y + h, x:x + w] = blurred_face

        # Draw UI text and tracking rectangles on top of the blur
        # 1. Dataset Counter (Top Left)
        cv2.putText(frame, f"Count: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

        # 2. Gender and Age Label (Above the tracking square)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Outer Tracking Square
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
