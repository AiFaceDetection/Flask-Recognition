from flask import Flask, render_template, Response, request, sessions, url_for, redirect
import cv2
import os
from faceIdentification import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)



camera = cv2.VideoCapture(1)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def main_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        cv2.imwrite(os.path.join(BASE_DIR , 'unknown.jpg'), frame)
        if not success:
            break
        else:            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



def identification():
    full_file_path = os.path.join(BASE_DIR , 'unknown.jpg')
    predictions = predict(full_file_path, model_path="trained_knn_model.clf")
    names = []
    for name, (top, right, bottom, left) in predictions:
        names.append(name)
    return names

@app.route('/')
def index():
    return render_template('index.html',  identify = [])
    
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/main_feed')
def main_feed():
    return Response(main_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/identi')
def identi():
    return render_template('index.html',  identify = identification())
    
if __name__ == '__main__':
    app.run(debug=True)