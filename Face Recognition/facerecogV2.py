import cv2
import os
import numpy as np
import csv
from datetime import datetime


# Initialize
dataset_path = "Dataset"

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" #for the internal files
face_cascade = cv2.CascadeClassifier(cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
names = {}
label_id = 0


# Load Dataset and Train
print("Loading dataset...")

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    names[label_id] = person_name

    for image_name in os.listdir(person_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')): #for the external files
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected_faces:
                #faces.append(gray[y:y+h, x:x+w])
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                faces.append(face_img)
                labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("No faces found in dataset.")
    exit()

recognizer.train(faces, np.array(labels))
print("Training Completed!")


# Attendance Function
marked_names = set()

def mark_attendance(name):
    if name in marked_names:
        return

    file_exists = os.path.isfile("Attendance.csv")

    with open("Attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        writer.writerow([name, date, time])

    marked_names.add(name)


# Start Camera
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        #face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face_roi)

        if confidence < 80: # Confidence threshold for recognition 60, 80, 90 (bases on the accuracy of the dataset need to be atleast 40-60 pictures for higher accuracy)
            name = names[label]
            mark_attendance(name)
            text = f"{name} ({round(confidence,1)})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()