'''
import cv2
import numpy as np

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/mitushijain/Desktop/Face_scanning code/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# 1. Extract the live face region from the current frame
face_roi = frame[y:y + h, x:x + w]

# 2. Apply a strong Gaussian blur to that specific region
# Note: (99, 99) is the kernel size. It must be positive and ODD. Higher numbers = stronger blur.
blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)

# 3. Paste the blurred region back into the frame
frame[y:y + h, x:x + w] = blurred_face
'''
import cv2
import numpy as np

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/mitushijain/Desktop/Face_scanning code/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Your existing dataset collection logic (saves CLEAN unblurred faces)
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1

        # === FACE BLURRING LOGIC ===
        face_roi = frame[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y + h, x:x + w] = blurred_face
        # ===========================

        # Draw UI elements on top of the blurred frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
