import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # In CelebA dataset, we don't have individual labels, so treating each image as a different person
            labels.append(os.path.basename(filename))  
    return images, labels

# Load training data (CelebA dataset)
celeba_folder = 'path/to/celeba_dataset'
X_train, y_train = load_images_from_folder(celeba_folder)

# Preprocess training data (resize, flatten)
X_train = [cv2.resize(img, (100, 100)).flatten() for img in X_train]

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize faces
def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100)).flatten()
        
        # Predict the label for the face
        label = knn.predict([face_roi])[0]
        
        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    return frame

# Main function
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = recognize_faces(frame)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
