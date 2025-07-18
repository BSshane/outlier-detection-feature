import os
import time
import logging
import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template
from datetime import datetime

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('./Dlib_face_recognition_from_camera/data/data_dlib/shape_predictor_68_face_landmarks.dat')
# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("./Dlib_face_recognition_from_camera/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- Flask 应用初始化 ---
app = Flask(__name__)


class FaceRecognizerService:
    def __init__(self):
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        self.features_known_list = []
        self.face_name_known_list = []

        # --- 识别逻辑参数 (改进点1) ---
        # 稍微放宽阈值，让已录入人脸更容易被识别。原为0.4，可调整为0.42-0.45之间
        self.recognition_threshold = 0.43

        # 用于存储当前帧的识别结果，以便在未检测的帧上继续显示
        self.current_frame_face_position_list = []
        self.current_frame_name_list = []

        # 帧处理间隔，每 DETECT_INTERVAL 帧检测一次
        self.detect_interval = 5
        self.frame_count = 0

        # --- 危险状态检测参数 (改进点2) ---
        # 当单帧检测到超过 DANGER_UNKNOWN_FACES_TRIGGER 个未知人脸时，开始计数
        self.DANGER_UNKNOWN_FACES_TRIGGER = 1 # 调整为2表示至少2个陌生人才算
        # 连续 DANGER_CONSECUTIVE_FRAMES_TRIGGER 帧满足条件，则触发危险状态
        self.DANGER_CONSECUTIVE_FRAMES_TRIGGER = 10

        self.consecutive_unknown_frames_count = 0
        self.is_danger_state = False

        # --- 危险状态响应参数 (改进点3) ---
        self.last_saved_time = 0
        self.save_interval = 5  # 每5秒最多保存一张快照

        # 创建用于保存快照的目录
        self.snapshots_dir = os.path.join(BASE_DIR, "FileGetter")
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
            logging.info(f"创建快照目录: {self.snapshots_dir}")

        # 加载人脸数据库
        self.load_face_database()

    def load_face_database(self):
        """从 "features_all.csv" 读取录入人脸特征"""
        if os.path.exists("./Dlib_face_recognition_from_camera/data/features_all.csv"):
            path_features_known_csv = "./Dlib_face_recognition_from_camera/data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                # 直接从 CSV 读取中文名
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if pd.isna(csv_rd.iloc[i][j]):
                        features_someone_arr.append(0)
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(np.array(features_someone_arr, dtype=float))
            logging.info("成功加载人脸数据库，共 %d 张人脸", len(self.features_known_list))
        else:
            logging.warning("警告: './Dlib_face_recognition_from_camera/data/features_all.csv' 未找到!")
            logging.warning("请先运行特征提取脚本生成该文件。")

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        """计算两个128D向量间的欧式距离"""
        dist = np.linalg.norm(feature_1 - feature_2)
        return dist

    def handle_danger_event(self, frame):
        """
        处理危险状态事件：保存快照并为数据库操作预留位置 (改进点3)
        """
        current_time = time.time()
        # 节流阀：防止短时间内保存大量图片
        if current_time - self.last_saved_time > self.save_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.snapshots_dir, f"danger_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            logging.warning(f"危险状态！已保存快照到: {filename}")

            # -------- TODO: 在此处添加数据库插入语句 --------
            # 例如:
            # db_connection = get_db_connection()
            # cursor = db_connection.cursor()
            # sql = "INSERT INTO alerts (timestamp, image_path, reason) VALUES (%s, %s, %s)"
            # cursor.execute(sql, (datetime.now(), filename, "检测到多个连续陌生人"))
            # db_connection.commit()
            # cursor.close()
            # db_connection.close()
            # ---------------------------------------------

            self.last_saved_time = current_time

    def process_and_recognize(self, img_rd):
        """
        处理单帧图像，进行人脸检测和识别
        """
        # 1. 检测人脸
        faces = detector(img_rd, 0)

        # 清空上一帧的结果
        self.current_frame_face_position_list = []
        self.current_frame_name_list = []

        unknown_faces_this_frame=0

        # 如果检测到人脸
        if len(faces) > 0:
            for  d in faces:
                # 2. 提取128D特征
                shape = predictor(img_rd, d)
                face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
                face_feature = np.array(face_descriptor)
                # 默认设为未知
                name = "Unknown"

                distances = [self.return_euclidean_distance(face_feature, known_feature) for known_feature in
                             self.features_known_list]
                # # 3. 与数据库比对
                # distances = []
                # for i in range(len(self.features_known_list)):
                #     dist = self.return_euclidean_distance(face_feature, self.features_known_list[i])
                #     distances.append(dist)

                # 找到最小距离
                if distances:
                    min_dist_index = np.argmin(distances)
                    min_dist = distances[min_dist_index]

                    # 阈值判断
                    if min_dist <self.recognition_threshold:
                        name = self.face_name_known_list[min_dist_index]
                        logging.info(f"识别成功: {name}, 距离: {min_dist:.2f}")
                    else:
                        unknown_faces_this_frame += 1  # 统计本帧的未知人
                        logging.info(f"识别到未知人脸, 最近距离: {min_dist:.2f}")
                else:
                    unknown_faces_this_frame += 1

                # 存储结果
                # 人脸位置 (用于绘制矩形框)
                pos = (d.left(), d.top(), d.right(), d.bottom())
                # 姓名位置 (用于绘制文字)
                text_pos = (d.left(), d.bottom() + 5)

                self.current_frame_face_position_list.append((pos, text_pos))
                self.current_frame_name_list.append(name)
        # 更新危险状态计数器 (改进点2)
        if unknown_faces_this_frame >= self.DANGER_UNKNOWN_FACES_TRIGGER:
            self.consecutive_unknown_frames_count += 1
        else:
            # 如果不满足条件，重置计数器和危险状态
            self.consecutive_unknown_frames_count = 0
            self.is_danger_state = False

        # 判断是否触发危险状态
        if self.consecutive_unknown_frames_count >= self.DANGER_CONSECUTIVE_FRAMES_TRIGGER:
            if not self.is_danger_state:
                 logging.warning(f"进入危险状态！连续 {self.consecutive_unknown_frames_count} 帧检测到 {unknown_faces_this_frame} 个陌生人。")
            self.is_danger_state = True

    def draw_results(self, img_rd):
        """
        在图像上绘制识别结果（矩形框和姓名）
        """
        # 使用 Pillow 绘制，支持中文
        img_pil = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for i in range(len(self.current_frame_name_list)):
            pos, text_pos = self.current_frame_face_position_list[i]
            name = self.current_frame_name_list[i]

            # 绘制矩形框
            draw.rectangle(pos, outline="green", width=2)
            # 绘制姓名
            draw.text(text_pos, name, font=self.font_chinese, fill=(0, 255, 0))

        # 如果处于危险状态，在画面上显示警告信息 (改进点2)
        if self.is_danger_state:
            # 添加一个半透明的红色背景条
            draw.rectangle([0, 0, img_pil.width, 40], fill=(255, 0, 0, 128))
            warning_font = ImageFont.truetype("simsun.ttc", 30)
            draw.text((10, 5), "!!! DANGER: UNKNOWN FACES DETECTED !!!", font=warning_font, fill="white")

        # 将 Pillow Image 转换回 OpenCV 格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def generate_frames(self, rtmp_url):
        """
        生成器函数，从RTMP流读取，处理并以JPEG格式yield帧
        """
        logging.info(f"开始连接RTMP流: {rtmp_url}")
        cap = cv2.VideoCapture(rtmp_url)

        if not cap.isOpened():
            logging.error("错误: 无法打开RTMP流！请检查URL和网络。")
            # 可以生成一张表示错误的图片
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error: Cannot open RTMP stream", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return

        while True:
            success, img_rd = cap.read()
            if not success:
                logging.warning("RTMP流结束或读取帧失败。")
                break

            self.frame_count += 1

            # 每隔 DETECT_INTERVAL 帧进行一次检测识别
            if self.frame_count % self.detect_interval == 0:
                # 这是“特定帧”，进行耗时的AI处理
                try:
                    self.process_and_recognize(img_rd)
                except Exception as e:
                    logging.error(f"处理帧时发生错误: {e}")

            # 如果处于危险状态，执行响应动作 (改进点3)
            if self.is_danger_state:
                self.handle_danger_event(img_rd)
            # 在每一帧上都绘制最新的识别结果
            processed_img = self.draw_results(img_rd)

            # 将处理后的帧编码为JPEG
            ret, buffer = cv2.imencode('.jpg', processed_img)
            if not ret:
                logging.warning("JPEG 编码失败")
                continue

            frame = buffer.tobytes()

            # 使用 multipart/x-mixed-replace 格式 yield 帧
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        logging.info("RTMP流连接已关闭。")

# --- 创建服务实例 (全局) ---
face_service = FaceRecognizerService()

# --- Flask 路由 ---
@ app.route('/')
def index():
    """提供前端页面"""
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由，返回一个multipart response"""
    # 在这里填写你的RTMP服务器地址
    # 例如： "rtmp://192.168.1.100/live/stream"
    RTMP_URL = "rtmp://1.92.135.70:9090/live/1"

    return Response(face_service.generate_frames(RTMP_URL),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- 主程序入口 ---
if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 启动Flask应用，host='0.0.0.0' 使其可以被局域网访问
    app.run(host='0.0.0.0', port=5000, debug=True)