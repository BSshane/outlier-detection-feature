import os
import cv2
import dlib
import numpy as np
import pandas as pd
import subprocess
import logging
import threading
from PIL import Image, ImageDraw, ImageFont
from flask import Flask
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)

# Dlib 模型加载
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()


class FaceRecognizerOT:
    def __init__(self):
        self.font = ImageFont.truetype("simsun.ttc", 25)
        self.tracked_faces = []
        self.next_face_id = 0
        self.max_dist_for_tracking = 50
        self.max_lost_frames = 5
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_face_database()

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                features = [float(x) if x != '' else 0.0 for x in csv_rd.iloc[i][1:129]]
                self.face_features_known_list.append(np.array(features))
            logging.info(f"已加载 {len(self.face_name_known_list)} 张已知人脸")
        else:
            logging.warning("未找到 features_all.csv，跳过已知人脸加载")

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(f1 - f2)

    def recognize_face(self, img, rect):
        shape = predictor(img, rect)
        feature = np.array(face_reco_model.compute_face_descriptor(img, shape))
        min_dist = float("inf")
        name = "unknown"
        for i, known in enumerate(self.face_features_known_list):
            dist = self.euclidean_distance(feature, known)
            if dist < min_dist:
                min_dist = dist
                name = self.face_name_known_list[i] if dist < 0.4 else "unknown"
        return name, feature

    def draw_chinese_name(self, img, names, positions):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for name, pos in zip(names, positions):
            draw.text((int(pos[0]), int(pos[1])), name, font=self.font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process_single_frame(self, img):
        orig_h, orig_w = img.shape[:2]
        scale = 320 / orig_w
        small_img = cv2.resize(img, (320, int(orig_h * scale)))
        faces = detector(small_img, 0)

        current_centroids = []
        current_rects = []

        for face in faces:
            l, t, r, b = [int(v / scale) for v in [face.left(), face.top(), face.right(), face.bottom()]]
            rect = dlib.rectangle(l, t, r, b)
            current_rects.append(rect)
            current_centroids.append(((l + r) // 2, (t + b) // 2))

        for f in self.tracked_faces:
            f['lost_cnt'] += 1

        cost_matrix = [[np.linalg.norm(np.array(c) - np.array(f['centroid'])) for f in self.tracked_faces] for c in current_centroids]
        row_ind, col_ind = linear_sum_assignment(cost_matrix) if cost_matrix else ([], [])

        assigned = set()
        new_tracked = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < self.max_dist_for_tracking:
                f = self.tracked_faces[c]
                f['centroid'] = current_centroids[r]
                f['lost_cnt'] = 0
                f['bbox'] = current_rects[r]
                new_tracked.append(f)
                assigned.add(c)

        for idx in set(range(len(current_centroids))) - set(row_ind):
            rect = current_rects[idx]
            name, feature = self.recognize_face(img, rect)
            new_tracked.append({
                'id': self.next_face_id,
                'centroid': current_centroids[idx],
                'name': name,
                'feature': feature,
                'lost_cnt': 0,
                'bbox': rect
            })
            self.next_face_id += 1

        for idx in set(range(len(self.tracked_faces))) - assigned:
            f = self.tracked_faces[idx]
            if f['lost_cnt'] <= self.max_lost_frames:
                new_tracked.append(f)

        self.tracked_faces = new_tracked

        for f in self.tracked_faces:
            l, t, r, b = f['bbox'].left(), f['bbox'].top(), f['bbox'].right(), f['bbox'].bottom()
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
            img = self.draw_chinese_name(img, [f['name']], [(l, t - 30)])

        return img


recognizer = FaceRecognizerOT()

def log_ffmpeg_stderr(proc):
    for line in proc.stderr:
        print("FFmpeg:", line.decode(errors='ignore').strip())

def process_and_stream(input_url, output_url):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '640x480',
        '-r', '25',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        output_url
    ]

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    threading.Thread(target=log_ffmpeg_stderr, args=(ffmpeg_proc,), daemon=True).start()

    cap = cv2.VideoCapture(input_url)
    if not cap.isOpened():
        logging.error(f"无法打开输入流: {input_url}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = recognizer.process_single_frame(frame)
            resized = cv2.resize(processed, (640, 480))
            ffmpeg_proc.stdin.write(resized.tobytes())
    except Exception as e:
        logging.error(f"流处理错误: {e}")
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()


@app.route('/start_processing/<stream_id>')
def start_processing(stream_id):
    input_url = f'rtmp://1.92.135.70:9090/live/{stream_id}'
    output_url = f'rtmp://1.92.135.70:9090/live/processed_{stream_id}'

    threading.Thread(
        target=process_and_stream,
        args=(input_url, output_url),
        daemon=True
    ).start()

    return {
        'status': 'processing_started',
        'input': input_url,
        'output': output_url
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000)
