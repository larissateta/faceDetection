import cv2

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error opening video stream or file")


while True:
    ret, frame = video_capture.read()
    if ret:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Loop through all the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face and save it to a file .jpg
            face = frame[y:y+h, x:x+w]
            cv2.imwrite('face.jpg', face)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Wait for key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


video_capture.release()
cv2.destroyAllWindows()
