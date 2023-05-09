import cv2
import os

input_image_path = "input.jpg"
input_image = cv2.imread(input_image_path)

# Create a CascadeClassifier object for face detection
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the input image
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Crop and save each detected face
for i, (x, y, w, h) in enumerate(faces):
    face_image = input_image[y:y+h, x:x+w]
    output_image_path = f"face_{i}.jpg"
    cv2.imwrite(output_image_path, face_image)
    print(f"Saved face {i} to {output_image_path}")

print("Done")
