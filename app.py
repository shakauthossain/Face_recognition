from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import numpy as np
import requests
import datetime
import pickle
import json

app = Flask(__name__)

# Load the trained face encodings and names
def load_trained_faces(filename='trained_faces.pkl'):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_trained_faces()

def recognize_faces(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        recognized_names.append(name)

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label with name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, recognized_names

def mark_attendance(names):
    attendance_log = []
    now = datetime.datetime.now()
    for name in names:
        attendance_log.append({'name': name, 'time': now.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"Attendance marked for {name} at {now}")
    return attendance_log

@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame_count += 1
            if frame_count % frame_skip == 0:
                frame, recognized_names = recognize_faces(frame, known_face_encodings, known_face_names)
                attendance_log = mark_attendance(recognized_names)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
            # To save or upload the attendance log, you can call your upload function here

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_attendance', methods=['POST'])
def upload_attendance():
    # This route can be used to receive attendance logs from the front-end if needed
    attendance_log = request.json.get('attendance_log')
    print(f"Received attendance log: {attendance_log}")
    # Process the attendance log as needed
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
