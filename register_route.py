import os
from app import app, request, camera, cv2, render_template, Response, BASE_DIR, image_dir
from train_data import train

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
 
@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    count = request.form['count']
    path = "./images/"+name+"/"
    try:
        os.makedirs(path)
    except:
        pass
    _, frame = camera.read()
    cv2.imwrite(f'{path+name+count}.jpg', frame)
    return Response(status = 200)

@app.route('/train_model', methods = ['POST'])
def train_model():
    print('Training')
    train(image_dir, os.path.join(BASE_DIR, "trained_knn_model.clf"), n_neighbors=2)
    return render_template('register.html', message='Training Complete')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')