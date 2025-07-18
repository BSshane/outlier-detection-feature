from flask import Flask, Response, render_template
import cv2
import dlib
import numpy as np
import os
import logging
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from collections import deque

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from datetime import datetime
import pymysql
import threading
import queue
import time

app = Flask(__name__)
#cd Dlib_face_recognition_from_camera
# --- dlib模型路径 ---
predictor_path = './Dlib_face_recognition_from_camera/data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_reco_model_path = './Dlib_face_recognition_from_camera/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

# --- 简易3DCNN活体检测模型 ---
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
liveness_model = Simple3DCNN().to(device)
liveness_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def preprocess_face_frame(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    tensor = transform(face_resized)
    return tensor

# --- 人脸识别与告警逻辑封装 ---
class FaceRecognizerOT:
    def __init__(self):
        # 数据库连接
        self.conn = pymysql.connect(
            host='1.92.135.70',
            port=3306,
            user='root',
            password='Aa123321',
            database='AbnormalDetection',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.conn.cursor()

        self.font = ImageFont.truetype("simsun.ttc", 25)

        self.tracked_faces = []
        self.next_face_id = 0
        self.max_dist_for_tracking = 50
        self.max_lost_frames = 5
        self.seq_len = 16

        self.face_features_known_list = []
        self.face_name_known_list = []

        self.load_face_database()

        # 异步告警队列与线程
        self.warning_queue = queue.Queue()
        self.warning_thread = threading.Thread(target=self.warning_worker, daemon=True)
        self.warning_thread.start()

        # 告警图片保存目录
        self.alert_img_dir = "alert_images"
        os.makedirs(self.alert_img_dir, exist_ok=True)

    def load_face_database(self):
        sql = "SELECT name, " + ','.join([f"x{i+1}" for i in range(128)]) + " FROM face"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        if not results:
            logging.warning("数据库中无人脸数据")
            return

        self.face_name_known_list.clear()
        self.face_features_known_list.clear()
        for row in results:
            self.face_name_known_list.append(row['name'])
            feature = np.array([row[f"x{i+1}"] for i in range(128)], dtype=np.float32)
            self.face_features_known_list.append(feature)
        logging.info(f"已从数据库加载 {len(self.face_name_known_list)} 张已知人脸")

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(f1 - f2)

    def recognize_face(self, img, rect):
        shape = predictor(img, rect)
        feature = np.array(face_reco_model.compute_face_descriptor(img, shape))
        min_dist = float("inf")
        name = "unknown"
        for i, known_feature in enumerate(self.face_features_known_list):
            dist = self.euclidean_distance(feature, known_feature)
            if dist < min_dist:
                min_dist = dist
                if dist < 0.4:
                    name = self.face_name_known_list[i]
                else:
                    name = "unknown"
        return name, feature

    def detect_liveness(self, frames_deque):
        if len(frames_deque) < self.seq_len:
            return True
        tensor_seq = torch.stack(list(frames_deque), dim=1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = liveness_model(tensor_seq)
            pred = torch.softmax(logits, dim=1)[0]
            return pred[1].item() > 0.7

    def draw_chinese_name(self, img, names, positions, color=(0, 255, 0)):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for name, pos in zip(names, positions):
            draw.text((int(pos[0]), int(pos[1])), name, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def save_alert_image(self, img, camera_id, alert_type):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert_type}_cam{camera_id}_{ts}.jpg"
        self.alert_img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FileGetter")
        filepath = os.path.join(self.alert_img_dir, filename)
        cv2.imwrite(filepath, img)
        return filename  # 返回文件名用于写数据库info字段

    def warning_worker(self):
        while True:
            try:
                camera_id, warning_type, img_name = self.warning_queue.get()
                img_name = "http://127.0.0.1:9081/" + img_name
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                video_url = f'rtmp://1.92.135.70:9090/live/{camera_id}'
                sql = "INSERT INTO warning (cameraId, curTime, type, videoURL, info) VALUES (%s, %s, %s, %s, %s)"
                try:
                    self.cursor.execute(sql, (camera_id, now, warning_type, img_name, video_url))
                    self.conn.commit()
                    logging.info(f"[写入告警] 类型: {warning_type}, 摄像头: {camera_id}, 图片: {img_name}")
                except Exception as e:
                    logging.error(f"异步插入告警失败: {e}")
            except Exception as e:
                logging.error(f"告警线程异常: {e}")
            time.sleep(0.01)

    def insert_warning(self, camera_id, warning_type, img_name):
        self.warning_queue.put((camera_id, warning_type, img_name))

    def process_single_frame(self, img, stream_url):
        camera_id = int(stream_url.split('/')[-1])
        orig_h, orig_w = img.shape[:2]
        target_w = 320
        scale = target_w / orig_w
        small_img = cv2.resize(img, (target_w, int(orig_h * scale)))
        faces = detector(small_img, 0)

        current_centroids = []
        current_rects = []

        for face in faces:
            left = int(face.left() / scale)
            top = int(face.top() / scale)
            right = int(face.right() / scale)
            bottom = int(face.bottom() / scale)
            rect = dlib.rectangle(left, top, right, bottom)
            current_rects.append(rect)
            current_centroids.append(((left + right) // 2, (top + bottom) // 2))

        for f in self.tracked_faces:
            f['lost_cnt'] += 1

        cost_matrix = []
        for c in current_centroids:
            row = [np.linalg.norm(np.array(c) - np.array(t['centroid'])) for t in self.tracked_faces]
            cost_matrix.append(row)
        if cost_matrix and cost_matrix[0]:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = ([], [])

        assigned = set()
        new_tracked_faces = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < self.max_dist_for_tracking:
                t = self.tracked_faces[c]
                t['centroid'] = current_centroids[r]
                t['lost_cnt'] = 0
                t['bbox'] = current_rects[r]

                left, top, right, bottom = t['bbox'].left(), t['bbox'].top(), t['bbox'].right(), t['bbox'].bottom()
                face_img = img[top:bottom, left:right]
                if face_img.size != 0:
                    tensor_face = preprocess_face_frame(face_img)
                    if 'frames_deque' not in t:
                        t['frames_deque'] = deque(maxlen=self.seq_len)
                    t['frames_deque'].append(tensor_face)

                is_live = self.detect_liveness(t['frames_deque'])
                t['is_live'] = is_live
                name, _ = self.recognize_face(img, current_rects[r])
                t['name'] = name

                # 准备带框和标签的告警图
                frame_for_alert = img.copy()
                box_color = (0, 0, 255) if not is_live else (0, 255, 0)
                label = f"{name}[欺骗]" if t.get('alert_type') == 'cheat' else name
                cv2.rectangle(frame_for_alert, (left, top), (right, bottom), box_color, 2)
                frame_for_alert = self.draw_chinese_name(frame_for_alert, [label], [(left, top - 30)], box_color)

                # 连续3次识别才报警逻辑
                if not is_live:
                    t['spoof_cnt'] = t.get('spoof_cnt', 0) + 1
                    t['unknown_cnt'] = 0  # 活体，重置陌生人计数
                    if t['spoof_cnt'] >= 3 and not t.get('alerted'):
                        img_name = self.save_alert_image(frame_for_alert, camera_id, 'cheat')
                        self.insert_warning(camera_id, 'cheat', img_name)
                        t['alert_type'] = 'cheat'
                        t['alerted'] = True
                else:
                    t['spoof_cnt'] = 0
                    if name == 'unknown':
                        t['unknown_cnt'] = t.get('unknown_cnt', 0) + 1
                        if t['unknown_cnt'] >= 3 and not t.get('alerted'):
                            img_name = self.save_alert_image(frame_for_alert, camera_id, 'stranger')
                            self.insert_warning(camera_id, 'stranger', img_name)
                            t['alert_type'] = 'stranger'
                            t['alerted'] = True
                    else:
                        t['unknown_cnt'] = 0

                new_tracked_faces.append(t)
                assigned.add(c)

        # 新检测到的人脸，未匹配已有轨迹
        for idx in set(range(len(current_centroids))) - set(row_ind):
            rect = current_rects[idx]
            name, feature = self.recognize_face(img, rect)
            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
            face_img = img[top:bottom, left:right]
            frames_deque = deque(maxlen=self.seq_len)
            if face_img.size != 0:
                frames_deque.append(preprocess_face_frame(face_img))
            new_tracked_faces.append({
                'id': self.next_face_id,
                'centroid': current_centroids[idx],
                'name': name,
                'feature': feature,
                'lost_cnt': 0,
                'bbox': rect,
                'frames_deque': frames_deque,
                'is_live': True,
                'alerted': False,
                'spoof_cnt': 0,
                'unknown_cnt': 1 if name == 'unknown' else 0,
                'alert_type': None
            })
            self.next_face_id += 1

        # 未匹配上的已追踪人脸，若未超时继续保留
        for idx in set(range(len(self.tracked_faces))) - assigned:
            t = self.tracked_faces[idx]
            if t['lost_cnt'] <= self.max_lost_frames:
                new_tracked_faces.append(t)

        self.tracked_faces = new_tracked_faces

        # 画所有框和标签（这里绘制最新状态）
        for t in self.tracked_faces:
            left, top, right, bottom = t['bbox'].left(), t['bbox'].top(), t['bbox'].right(), t['bbox'].bottom()
            color = (0, 0, 255) if not t['is_live'] else (0, 255, 0)
            label = f"{t['name']}[欺骗]" if t.get('alert_type') == 'cheat' else t['name']
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            img = self.draw_chinese_name(img, [label], [(left, top - 30)], color)

        return img

    def draw_tracked_faces_only(self, img):
        for t in self.tracked_faces:
            if 'bbox' in t:
                left, top, right, bottom = t['bbox'].left(), t['bbox'].top(), t['bbox'].right(), t['bbox'].bottom()
                color = (0, 255, 0) if t['is_live'] else (0, 0, 255)
                label = t['name'] if t['is_live'] else "[欺骗]"
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                img = self.draw_chinese_name(img, [label], [(left, top - 30)], color)
        return img

recognizer = FaceRecognizerOT()

def generate_video_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频流: {stream_url}")
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            frame = recognizer.process_single_frame(frame, stream_url)
        else:
            frame = recognizer.draw_tracked_faces_only(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<stream_id>')
def video_feed(stream_id):
    stream_url = f'rtmp://1.92.135.70:9090/live/{stream_id}'
    return Response(generate_video_stream(stream_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)