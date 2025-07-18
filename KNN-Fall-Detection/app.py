from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

app = Flask(__name__)

pose_knn = joblib.load('Model/PoseKeypoint.joblib')
keyXYZ = [
    f"{name}_{axis}"
    for name in [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    for axis in ["x", "y", "z"]
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    rtmp_url = 'rtmp://1.92.135.70:9090/live/1'
    cap = cv2.VideoCapture(rtmp_url)

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        prevTime = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                res_point = []
                for lm in results.pose_landmarks.landmark:
                    res_point.extend([lm.x, lm.y, lm.z])
                if len(res_point) == len(keyXYZ):
                    sample = np.array(res_point).reshape(1, -1)
                    pred = pose_knn.predict(sample)[0]
                    label = "Fall" if pred == 0 else "Normal"
                    color = (0, 0, 255) if pred == 0 else (0, 255, 0)
                    cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            currTime = time.time()
            fps = 1 / (currTime - prevTime) if prevTime != 0 else 0
            prevTime = currTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)