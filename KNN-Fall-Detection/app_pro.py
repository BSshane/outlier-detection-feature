from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import math
from mediapipe.framework.formats import landmark_pb2

app = Flask(__name__)

# 加载训练好的KNN模型
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


def calculate_angle(p1, p2):
    """计算两个点形成的向量与水平线的夹角，单位度"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(angle_rad * 180.0 / math.pi)
    return angle_deg


def is_fall(pose_landmarks, img_height):
    """
    结合机器学习结果与简单规则判断是否跌倒
    参数：
      pose_landmarks: MediaPipe姿态关键点（landmark对象列表）
      img_height: 帧图像高度（用于换算坐标）
    返回：
      True表示检测到跌倒，False表示正常
    """
    # 提取肩膀和臀部关键点坐标（归一化）
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # 计算躯干中心点坐标（x,y）
    torso_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
    torso_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

    # 计算躯干倾斜角度（肩膀中心与臀部中心）
    shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
    hip_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
    angle = calculate_angle(shoulder_center, hip_center)  # 0-180度

    # 计算鼻子高度（归一化y坐标转像素）
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
    nose_y_px = nose.y * img_height

    # 简单规则阈值（可调节）
    angle_threshold = 45   # 躯干倾斜大于45度视为异常
    height_threshold = img_height * 0.4  # 鼻子高度低于40%图高视为跌倒

    # 机器学习预测（假设样本准备好）
    res_point = []
    for lm in pose_landmarks:
        res_point.extend([lm.x, lm.y, lm.z])
    sample = np.array(res_point).reshape(1, -1)
    ml_pred = pose_knn.predict(sample)[0]  # 0:跌倒，1:正常

    # 结合判断
    if ml_pred == 0 and (angle > angle_threshold or nose_y_px > height_threshold):
        return True
    return False


def generate_frames():
    rtmp_url = 'rtmp://1.92.135.70:9090/live/1'
    cap = cv2.VideoCapture(rtmp_url)

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_count = 0
        last_landmarks = None
        last_fall_flag = False

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_count += 1

            orig_h, orig_w = frame.shape[:2]

            # 直接用原始帧做检测
            if frame_count % 3 == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    last_landmarks = results.pose_landmarks.landmark
                    last_fall_flag = is_fall(last_landmarks, orig_h)
                else:
                    last_landmarks = None
                    last_fall_flag = False

            image = frame.copy()

            if last_landmarks:
                landmarks_scaled = landmark_pb2.NormalizedLandmarkList()
                for lm in last_landmarks:
                    new_lm = landmark_pb2.NormalizedLandmark()
                    new_lm.x = lm.x
                    new_lm.y = lm.y
                    new_lm.z = lm.z
                    new_lm.visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    landmarks_scaled.landmark.append(new_lm)

                mp_drawing.draw_landmarks(
                    image,
                    landmarks_scaled,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                label = "Fall" if last_fall_flag else "Normal"
                color = (0, 0, 255) if last_fall_flag else (0, 255, 0)
                cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            ret, buffer = cv2.imencode('.jpg', image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)