from tkinter.constants import E
from flask import Flask, render_template, Response, request, sessions, url_for, redirect
from faceIdentification import predict
from faceComparison import compare
import cv2
import os

app = Flask(__name__)

camera = cv2.VideoCapture(1)  # use 0 for web camera
camera.set(3,1920)

HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")

face_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")

def main_frames():
    while True:
        success, frame = camera.read()  # read the camera frame

        card_frame = frame[HEIGHT-int(HEIGHT//1.4):int(HEIGHT//1.4), 0+20:int(40 * WIDTH // 100)-20]
        face_frame = frame[0+20:HEIGHT-20, int(40 * WIDTH // 100)+20: int(40 * WIDTH // 100) + WIDTH - int(40 * WIDTH // 100)-20]

        card_frame = rescale_frame(card_frame, percent=200)

        # card
        cv2.imwrite(os.path.join(card_dir , 'card.jpg'), card_frame)
        # face
        cv2.imwrite(os.path.join(face_dir , 'face.jpg'), face_frame)
        
        # Mask for CARD
        cv2.rectangle(frame, (0+10, int(HEIGHT//1.4)), (int(40 * WIDTH // 100)-10, HEIGHT-int(HEIGHT//1.4)), (255, 255, 255), 4)
        # Mask for FACE
        cv2.rectangle(frame, (int(40 * WIDTH // 100)+10, 0+10), (WIDTH-10, HEIGHT-10), (255, 255, 255), 4)
        
        if not success:
            break
        else:        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def identification():
    full_file_path = os.path.join(face_dir , 'face.jpg')
    predictions = predict(full_file_path, model_path="trained_knn_model.clf")
    names = []
    try:
        for name, (top, right, bottom, left) in predictions:
            names.append(name)
    finally:
        if len(names) == 0:
            names= ["Face not detected"]
        return names

def comparison():
    reasult = compare()
    return reasult

def rescale_frame (frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100) 
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)

@app.route('/')
def index():
    return render_template('index.html', identify = [], compareRea=[])

@app.route('/main_feed')
def main_feed():
    return Response(main_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/identi')
def identi():
    return render_template('index.html', identify = identification(), compareRea = comparison())


from register_route import *
    
if __name__ == '__main__':
    app.run(debug=True)