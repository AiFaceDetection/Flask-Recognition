from tkinter.constants import E
from flask import Flask, render_template, Response, request, sessions, url_for, redirect
from faceIdentification import predict
from faceComparison import compare
import cv2
import os

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
camera.set(3,1920)

HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")


def main_frames():
    while True:
        success, frame = camera.read()  # read the camera frame

        # card
        cv2.imwrite(os.path.join(card_dir , 'card.jpg'), frame[HEIGHT-int(HEIGHT//1.5):int(HEIGHT//1.5), 0+20:int(40 * WIDTH // 100)-20])
        # face
        cv2.imwrite(os.path.join(face_dir , 'face.jpg'), frame[0+20:HEIGHT-20, int(40 * WIDTH // 100)+20: int(40 * WIDTH // 100) + WIDTH - int(40 * WIDTH // 100)-20])
        
        # Mask for CARD
        cv2.rectangle(frame, (0+10, int(HEIGHT//1.5)), (int(40 * WIDTH // 100)-10, HEIGHT-int(HEIGHT//1.5)), (255, 255, 255), 4)
        # Mask for FACE
        cv2.rectangle(frame, (int(40 * WIDTH // 100)+10, 0+10), (WIDTH-10, HEIGHT-10), (255, 255, 255), 4)
        
        if not success:
            break
        else:        
            
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