import cv2
import numpy as np
import sys  # Added to handle clean script exiting

# 1. Initialize the Webcam and Face Detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/mitushijain/Desktop/Face_scanning code/haarcascade_frontalface_default.xml')

# 2. Initialize the Modern ONNX Network (Age Only)
AGE_ONNX = "/Users/mitushijain/Desktop/Face_scanning code/age_googlenet.onnx"
age_net = cv2.dnn.readNetFromONNX(AGE_ONNX)

# Technical constants required by the GoogLeNet model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# === EXIT CONSTANTS ===
last_predicted_age = None
stable_frame_count = 0
REQUIRED_STABLE_FRAMES = 10  # Must see the same age for 10 frames straight to exit
# ======================

faces_data = []
i = 0

print("🎥 Webcam active. Scanning face for age confirmation...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]

        if crop_img.size == 0:
            continue

        # Your dataset collection logic
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1

        # === DYNAMIC BOX EXPANSION ===
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        expanded_face = frame[y1:y2, x1:x2]

        # === DEEP LEARNING ANALYSIS LOGIC ===
        try:
            blob = cv2.dnn.blobFromImage(
                expanded_face,
                scalefactor=1.0,
                size=(227, 227),
                mean=MODEL_MEAN_VALUES,
                swapRB=True,
                crop=False
            )

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_index = age_preds.argmax()
            predicted_age = AGE_BUCKETS[age_index]
            label_text = f"Age: {predicted_age}"

            # === STABLE CONFIDENCE CHECK & TERMINAL EXIT ===
            if predicted_age == last_predicted_age:
                stable_frame_count += 1
            else:
                last_predicted_age = predicted_age
                stable_frame_count = 0  # Reset if the reading fluctuates

            if stable_frame_count >= REQUIRED_STABLE_FRAMES:
                print("\n===============================")
                print(f"🎯 SUCCESSFUL DETECTION!")
                print(f"Final Confirmed Age Range: {predicted_age}")
                print("===============================\n")

                # Cleanup camera threads and close app completely
                video.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            # ===============================================

        except Exception as e:
            label_text = "Analysis Failed"

        # === FACE BLURRING LOGIC ===
        face_roi = frame[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y + h, x:x + w] = blurred_face

        # Draw UI layout graphics
        cv2.putText(frame, f"Count: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
