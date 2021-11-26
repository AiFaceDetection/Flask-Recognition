from app import app, request, camera, cv2, render_template, Response


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
    print(name)
    _, frame = camera.read()
    cv2.imwrite(f'{name}.jpg', frame)
    return Response(status = 200)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')