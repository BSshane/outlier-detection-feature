# 调用流视频的脚本
import argparse

from flask import Flask, Response, render_template
import cv2
import dlib
import numpy as np
import pandas as pd
import os
import logging
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from collections import deque
from colorama import Fore, Style, init as colorama_init
import torch
import torch.nn as nn
import torchvision.transforms as transforms


import sys
from datetime import datetime
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，这样 INFO 和 DEBUG 消息就能被看到了
    format='[%(asctime)s][%(levelname)s][Alg-B] %(message)s', # 定义输出格式，更清晰
    stream=sys.stderr  # 明确指定输出到 stderr，虽然这是默认的
)

# dlib 模型加载
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'D:/1_MyPrograms/7SummerPrac/GiteeProject/AbnormalDetection/Video_Monitoring_Project/Dlib_face_recognition_from_camera/data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1(r"D:/1_MyPrograms/7SummerPrac/GiteeProject/AbnormalDetection/Video_Monitoring_Project/dlib_face_recognition_from_camera/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

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
        self.fc = nn.Linear(32, 2)  # 2分类：活体/欺骗

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
liveness_model = Simple3DCNN().to(device)
liveness_model.eval()  # 评估模式

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
def preprocess_face_frame(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    tensor = transform(face_resized)  # [3,112,112]
    return tensor

class FaceRecognizerOT:
    def __init__(self,detector,predictor,face_reco_model):
        try:
            self.font = ImageFont.truetype("simsun.ttc", 25)
        except IOError:
            logging.warning("simsun.ttc not found. Using default font.")
            self.font = ImageFont.load_default()

        self.detector = detector
        self.predictor = predictor
        self.face_reco_model = face_reco_model
        self.tracked_faces = []  # [{'id', 'centroid', 'name', 'feature', 'lost_cnt', 'bbox', 'frames_deque', 'is_live', 'alerted', 'spoof_cnt'}]
        self.next_face_id = 0
        self.max_dist_for_tracking = 50
        self.max_lost_frames = 5
        self.seq_len = 16  # 活体检测用帧序列长度

        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_face_database()

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    features.append(float(csv_rd.iloc[i][j]) if csv_rd.iloc[i][j] != '' else 0.0)
                self.face_features_known_list.append(np.array(features))
            logging.info(f"已加载 {len(self.face_name_known_list)} 张已知人脸")
        else:
            logging.warning("找不到 features_all.csv 文件，已知人脸库为空")

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

    def draw_chinese_name(self, img, names, positions, color=(0, 255, 0)):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for name, pos in zip(names, positions):
            draw.text((int(pos[0]), int(pos[1])), name, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def detect_liveness(self, frames_deque):
        if len(frames_deque) < self.seq_len:
            return True  # 未满帧，默认活体

        tensor_seq = torch.stack(list(frames_deque), dim=1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = liveness_model(tensor_seq)
            pred = torch.softmax(logits, dim=1)[0]
            live_prob = pred[1].item()
        return live_prob > 0.7

    def process_single_frame(self, img):
        orig_h, orig_w = img.shape[:2]
        target_w = 320
        scale = target_w / orig_w
        target_h = int(orig_h * scale)
        small_img = cv2.resize(img, (target_w, target_h))

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
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            current_centroids.append((cx, cy))

        for f in self.tracked_faces:
            f['lost_cnt'] += 1

        cost_matrix = []
        for c_cen in current_centroids:
            row = []
            for tracked in self.tracked_faces:
                dist = np.linalg.norm(np.array(c_cen) - np.array(tracked['centroid']))
                row.append(dist)
            cost_matrix.append(row)

        if len(cost_matrix) > 0 and len(cost_matrix[0]) > 0:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        assigned_tracked_ids = set()
        new_tracked_faces = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_dist_for_tracking:
                tracked = self.tracked_faces[c]
                tracked['centroid'] = current_centroids[r]
                tracked['lost_cnt'] = 0
                tracked['bbox'] = current_rects[r]

                left, top, right, bottom = current_rects[r].left(), current_rects[r].top(), current_rects[r].right(), current_rects[r].bottom()
                face_img = img[max(top,0):min(bottom,img.shape[0]), max(left,0):min(right,img.shape[1])]
                if face_img.size != 0:
                    tensor_face = preprocess_face_frame(face_img)
                    if 'frames_deque' not in tracked:
                        tracked['frames_deque'] = deque(maxlen=self.seq_len)
                    tracked['frames_deque'].append(tensor_face)

                is_live = self.detect_liveness(tracked.get('frames_deque', deque()))
                tracked['is_live'] = is_live

                # 连续欺骗计数逻辑
                if not is_live:
                    tracked['spoof_cnt'] = tracked.get('spoof_cnt', 0) + 1
                else:
                    tracked['spoof_cnt'] = 0

                # 连续3次欺骗才告警
                if tracked['spoof_cnt'] >= 3 and not tracked.get('alerted', False):
                    alert_msg = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}] SPOOF DETECTED ▶ ID={tracked['id']}  Name={tracked['name']}")
                    logging.warning(alert_msg)
                    sys.stdout.write(Fore.RED + alert_msg + Style.RESET_ALL + '\n')
                    sys.stdout.flush()
                    tracked['alerted'] = True

                new_tracked_faces.append(tracked)
                assigned_tracked_ids.add(c)

        unmatched_current_indices = set(range(len(current_centroids))) - set(row_ind)
        for idx in unmatched_current_indices:
            rect = current_rects[idx]
            name, feature = self.recognize_face(img, rect)

            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
            face_img = img[max(top,0):min(bottom,img.shape[0]), max(left,0):min(right,img.shape[1])]
            frames_deque = deque(maxlen=self.seq_len)
            if face_img.size != 0:
                tensor_face = preprocess_face_frame(face_img)
                frames_deque.append(tensor_face)

            is_live = True  # 默认活体

            new_tracked_faces.append({
                'id': self.next_face_id,
                'centroid': current_centroids[idx],
                'name': name,
                'feature': feature,
                'lost_cnt': 0,
                'bbox': rect,
                'frames_deque': frames_deque,
                'is_live': is_live,
                'alerted': False,
                'spoof_cnt': 0
            })
            self.next_face_id += 1

        unmatched_tracked_indices = set(range(len(self.tracked_faces))) - assigned_tracked_ids
        for idx in unmatched_tracked_indices:
            tracked = self.tracked_faces[idx]
            if tracked['lost_cnt'] <= self.max_lost_frames:
                new_tracked_faces.append(tracked)

        self.tracked_faces = new_tracked_faces

        for tracked in self.tracked_faces:
            bbox = tracked['bbox']
            left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
            color = (0, 255, 0) if tracked.get('is_live', True) else (0, 0, 255)
            label = f"{tracked['name']}[欺骗]" if not tracked.get('is_live', True) else tracked['name']
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            img = self.draw_chinese_name(img, [label], [(left, top - 30)], color=color)

        return img

    def draw_tracked_faces_only(self, img):
        for tracked in self.tracked_faces:
            bbox = tracked.get('bbox', None)
            if bbox is not None:
                left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
                color = (0, 255, 0) if tracked.get('is_live', True) else (0, 0, 255)
                label = tracked['name'] if tracked.get('is_live', True) else "[欺骗]"
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                img = self.draw_chinese_name(img, [label], [(left, top - 30)], color=color)
        return img


class StreamProcessor:
    """
    封装了所有模型加载和视频流处理的逻辑。
    """

    def __init__(self):
        logging.info("Initializing StreamProcessor...")
        # 1. 初始化模型 (所有昂贵的加载操作都在这里一次性完成)
        try:
            self.detector = detector
            # 注意：请确保这些数据文件路径相对于此脚本是正确的
            self.predictor = predictor
            self.face_reco_model = face_reco_model
            self.cnn_model = Simple3DCNN().to(device)
            # self.cnn_model.load_state_dict(...) # 如果有预训练权重，在这里加载
            self.cnn_model.eval()

            # 2. 初始化识别器业务逻辑类
            self.recognizer = FaceRecognizerOT(
                self.detector, self.predictor, self.face_reco_model
            )
            logging.info("All models loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            sys.exit(1)

    def generate_video_stream(self, stream_url: str, process_interval: int,output_dir: str):
        """
        核心处理函数，使用更健壮的 grab/retrieve 模式读取视频流。
        """
        print("[DEBUG] Initializing video stream...", file=sys.stderr, flush=True)
        try:
            source = int(stream_url)
            print(f"[INFO] Opening camera with index: {source}", file=sys.stderr, flush=True)
        except ValueError:
            source = stream_url
            print(f"[INFO] Opening video file or network stream: {source}", file=sys.stderr, flush=True)

        # 尝试使用 FFmpeg 后端，它通常更稳定
        # cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {stream_url}", file=sys.stderr, flush=True)
            try:
                build_info = cv2.getBuildInformation()
                video_io_line = [line for line in build_info.split('\n')if 'Vedio I/O'in line]
                print(f"[DEBUG] 4. OpenCV Video I/O backends: {video_io_line}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[DEBUG] 4. Could not get OpenCV build info: {e}", file=sys.stderr, flush=True)
            print("--- END DEBUGGING ---", file=sys.stderr, flush=True)
            return

        print("[INFO] Video source opened successfully. Starting frame loop.", file=sys.stderr, flush=True)
        frame_count = 0

        # --- 这是修改后的健壮循环 ---
        while True:
            # 步骤 1: 抓取帧到内存，这个操作很快，不易阻塞
            grabbed = cap.grab()

            if not grabbed:
                # 如果抓取失败，说明视频流结束或出错
                print("[INFO] End of video stream (grab failed).", file=sys.stderr, flush=True)
                break

            # (可选) 如果处理速度跟不上，可以在这里做跳帧
            # if frame_count % 2 != 0: # 每隔一帧处理一次
            #     frame_count += 1
            #     continue

            # 步骤 2: 从内存中解码帧，如果这一步失败，可以跳过，不影响下一次抓取
            retval, frame = cap.retrieve()
            if not retval:
                print("[WARNING] Failed to retrieve frame, skipping.", file=sys.stderr, flush=True)
                continue

            # --- 后续的处理逻辑完全不变 ---
            frame_count += 1

            process_frame=None
            if frame_count % process_interval == 0:
                process_frame = self.recognizer.process_single_frame(frame)
            else:
                process_frame = self.recognizer.draw_tracked_faces_only(frame)

            if process_frame is not None:
                timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
                filename =f"{timestamp}.jpg"
                output_path = os.path.join(output_dir, filename)

                cv2.imwrite(output_path, process_frame,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                if frame_count % 30 == 0:
                    logging.info(f"Saved {frame_count}.")
            # 为了调试，先不调用复杂的模型，只做简单处理
            # print(f"[DEBUG] Frame {frame_count}: Processing...", file=sys.stderr, flush=True)
            # if frame_count % process_interval == 0:
            #     frame = self.recognizer.process_single_frame(frame)
            # else:
            #     frame = self.recognizer.draw_tracked_faces_only(frame)

            # 调试阶段的临时处理：在帧上画一个点，确认视频流在走
            # cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
            #
            # # print(f"[DEBUG] Frame {frame_count}: Encoding to JPEG...", file=sys.stderr, flush=True)
            # ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # if not ret:
            #     print("[WARNING] Failed to encode frame to JPEG.", file=sys.stderr, flush=True)
            #     continue
            #
            # # print(f"[DEBUG] Frame {frame_count}: Yielding frame.", file=sys.stderr, flush=True)
            # yield (b'--frame\r'b'Content-Type: image/jpeg\r\r' + buffer.tobytes() + b'\r')

        print("[INFO] Releasing video capture...", file=sys.stderr, flush=True)
        cap.release()
        print("[INFO] Subprocess finished gracefully.", file=sys.stderr, flush=True)

            # --- 主程序入口 ---


def main():
    """
    主函数，负责解析命令行参数并启动流处理器。
    """
    # 初始化彩色终端和日志记录
    colorama_init()
    print(sys.executable)  # 查看当前 Python 解释器路径
    # 将日志输出到标准错误(stderr)，这样就不会污染标准输出(stdout)的JPEG流
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='[%(asctime)s][%(levelname)s][Alg-B] %(message)s')

    parser = argparse.ArgumentParser(description="人脸识别与活体检测命令行工具")
    parser.add_argument('--stream-url', type=str, required=True,
                        help="视频流地址 (例如: '0' 代表摄像头, '/path/to/video.mp4', 'rtmp://...')")
    parser.add_argument('--process-interval', type=int, default=5,
                        help="每隔N帧进行一次完整的识别处理")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="保存处理后图片的输出文件夹路径")
    args = parser.parse_args()

    # 创建处理器实例 (这将触发所有模型的加载)
    processor = StreamProcessor()

    logging.info(f"Starting to process video from '{args.stream_url}'")
    logging.info(f"Processing interval set to every {args.process_interval} frames.")

    if not os.path.exists(args.output_dir):
        logging.info(f"Output directory '{args.output_dir}' not found. Creating it.")
        os.makedirs(args.output_dir)
    try:
        # 获取生成器并迭代，将每一帧数据块写入标准输出
        processor.generate_video_stream(args.stream_url, args.process_interval,args.output_dir)


    except KeyboardInterrupt:
        logging.info("Process interrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during streaming: {e}")
    finally:
        logging.info("Stream processing finished.")


if __name__ == '__main__':
    main()