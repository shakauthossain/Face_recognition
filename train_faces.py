import face_recognition
import os
import pickle
import cv2

def train_faces(training_dir='dataset'):
    known_face_encodings = []
    known_face_names = []

    # Iterate over each person's directory in the training directory
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)

                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)

                    if len(face_locations) != 1:
                        print(f"Image {filename} in {person_name} contains {len(face_locations)} faces. Skipping...")
                        continue

                    # Get face encoding
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"Processed {filename} for {person_name}")
                    else:
                        print(f"No encodings found in {filename} in {person_name}. Skipping...")

    # Save the encodings and names to a file
    with open('trained_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("Training complete")

def preprocess_images(training_dir='dataset'):
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)

                    # Load image with OpenCV
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"Error loading image {filename} in {person_name}. Skipping...")
                        continue

                    # Convert to grayscale
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detect faces using Haar Cascade
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    if len(faces) == 0:
                        print(f"No faces found in {filename} in {person_name}. Skipping...")
                        continue

                    # Draw rectangle around the faces and save the preprocessed image
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # Save the preprocessed image
                    preprocessed_image_path = os.path.join(training_dir, 'preprocessed', person_name, filename)
                    os.makedirs(os.path.dirname(preprocessed_image_path), exist_ok=True)
                    cv2.imwrite(preprocessed_image_path, image)
                    print(f"Saved preprocessed image: {preprocessed_image_path}")

if __name__ == "__main__":
    preprocess_images()
    train_faces()