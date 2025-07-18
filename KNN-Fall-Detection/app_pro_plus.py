import os
import cv2
import math
import time
import queue
import joblib
import logging
import threading
import numpy as np
import mediapipe as mp
from datetime import datetime
from flask import Flask, render_template, Response
from mediapipe.framework.formats import landmark_pb2
import pymysql
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# 加载KNN模型
pose_knn = joblib.load('Model/PoseKeypoint.joblib')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 保存图片目录，放在项目同级目录FileGetter/img里
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE_DIR = os.path.dirname(BASE_DIR)  # 项目目录的上一级
IMG_SAVE_DIR = os.path.join(SAVE_BASE_DIR, 'FileGetter', 'img')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

warning_queue = queue.Queue(maxsize=100)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def get_db_connection():
    conn = pymysql.connect(
        host='1.92.135.70',
        port=3306,
        user='root',
        password='Aa123321',
        database='AbnormalDetection',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

# 载入中文字体
font_path = os.path.join(BASE_DIR, 'simsun.ttc')
font = ImageFont.truetype(font_path, 25)

class WarningWriter(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.warning_queue = warning_queue
        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()
        self.daemon = True

    def run(self):
        while True:
            try:
                camera_id, _, img_name = self.warning_queue.get()
                img_name = "http://127.0.0.1:9081/img/" + img_name
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                rtmp_url = f'rtmp://1.92.135.70:9090/live/{camera_id}'
                sql = "INSERT INTO warning (cameraId, curTime, type, videoURL, info) VALUES (%s, %s, %s, %s, %s)"
                warning_type = 'tumble'
                try:
                    self.cursor.execute(sql, (camera_id, now, warning_type, img_name, rtmp_url))
                    self.conn.commit()
                    logging.info(f"[写入告警] 类型: {warning_type}, 摄像头: {camera_id}, 图片: {img_name}")
                except Exception as e:
                    logging.error(f"异步插入告警失败: {e}")
            except Exception as e:
                logging.error(f"告警线程异常: {e}")
            time.sleep(0.01)

def put_text_chinese(img, text, position, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(angle_rad * 180.0 / math.pi)
    return angle_deg

def is_fall(pose_landmarks, img_height):
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
    hip_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
    angle = calculate_angle(shoulder_center, hip_center)

    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
    nose_y_px = nose.y * img_height

    angle_threshold = 45
    height_threshold = img_height * 0.4

    res_point = []
    for lm in pose_landmarks:
        res_point.extend([lm.x, lm.y, lm.z])
    sample = np.array(res_point).reshape(1, -1)
    ml_pred = pose_knn.predict(sample)[0]

    if ml_pred == 0 and (angle > angle_threshold or nose_y_px > height_threshold):
        return True
    return False

def generate_frames():
    rtmp_url = 'rtmp://1.92.135.70:9090/live/1'
    cap = cv2.VideoCapture(rtmp_url)

    process_width, process_height = 320, 180

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_count = 0
        fall_count = 0
        alerted = False
        tracked = False

        results = None  # 先定义，避免访问时报错

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtmp_url)
                continue

            frame_count += 1
            orig_h, orig_w = frame.shape[:2]
            small_frame = cv2.resize(frame, (process_width, process_height))

            fall_detected = False

            if frame_count % 3 == 0:
                image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                try:
                    results = pose.process(image_rgb)
                except Exception as e:
                    logging.error(f"pose.process异常: {e}")
                    results = None

                if results and results.pose_landmarks:
                    tracked = True
                    landmarks = results.pose_landmarks.landmark
                    fall_detected = is_fall(landmarks, orig_h)

                    if fall_detected:
                        fall_count += 1
                    else:
                        fall_count = 0

                    if fall_count >= 3 and not alerted:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_name = f'fall_cam1_{timestamp}.jpg'
                        img_path = os.path.join(IMG_SAVE_DIR, img_name)
                        cv2.imwrite(img_path, frame)
                        warning_queue.put(('1', 'tumble', img_name))
                        alerted = True
                        logging.info(f"报警: 摄像头1连续检测到摔倒, 图片: {img_name}")
                else:
                    if tracked:
                        logging.info("人员脱离跟踪，状态重置")
                    tracked = False
                    fall_count = 0
                    alerted = False

            image = frame.copy()
            if results and results.pose_landmarks:
                landmarks_scaled = landmark_pb2.NormalizedLandmarkList()
                for lm in results.pose_landmarks.landmark:
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
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                label = "跌倒" if fall_detected else "正常"
                color = (0, 0, 255) if fall_detected else (0, 255, 0)
                image = put_text_chinese(image, label, (30, 60), color=color)

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
    writer_thread = WarningWriter()
    writer_thread.start()
    app.run(debug=True, host='0.0.0.0', port=9089)