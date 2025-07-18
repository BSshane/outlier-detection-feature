import os
import time
import logging
import tkinter as tk
from tkinter import font as tkFont, messagebox
import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import pymysql

# Dlib模型路径
PREDICTOR_PATH = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECO_MODEL_PATH = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

# Dlib初始化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_reco_model = dlib.face_recognition_model_v1(FACE_RECO_MODEL_PATH)

# MySQL连接参数
MYSQL_CONFIG = {
    'host': '1.92.135.70',
    'port': 3306,
    'user': 'root',
    'password': 'Aa123321',
    'database': 'AbnormalDetection',
    'charset': 'utf8mb4'
}


class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.ss_cnt = 0
        self.registered_names = []
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC
        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # Tkinter GUI初始化
        self.win = tk.Tk()
        self.win.title("人脸录入系统")
        self.win.geometry("1300x550")

        # 左侧摄像头画面
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # 右侧信息面板
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text="0")
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info, width=25)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        # 字体设置
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.frame_right_info.pack()

        # 摄像头初始化
        self.cap = cv2.VideoCapture(0)
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # 数据库连接
        self.conn = None
        self.cursor = None
        self.connect_to_database()

        # 初始化工作
        self.pre_work_mkdir()
        self.check_existing_faces()
        self.GUI_info()

    def __del__(self):
        self.close_database_connection()
        if self.cap.isOpened():
            self.cap.release()

    def connect_to_database(self):
        """建立数据库连接"""
        try:
            self.conn = pymysql.connect(**MYSQL_CONFIG)
            self.cursor = self.conn.cursor()
            logging.info("数据库连接成功")
        except Exception as e:
            logging.error(f"数据库连接失败: {e}")
            messagebox.showerror("错误", "数据库连接失败，请检查配置和网络")

    def close_database_connection(self):
        """关闭数据库连接"""
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            logging.info("数据库连接已关闭")
        except Exception as e:
            logging.error(f"关闭数据库连接时出错: {e}")

    def pre_work_mkdir(self):
        """创建必要的目录"""
        if not os.path.isdir(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera)

    def check_existing_faces(self):
        """检查数据库中已存在的人脸数据（只统计数据库行数）"""
        self.registered_names = []

        # 仅从数据库加载已注册的人脸数量
        if self.conn and self.cursor:
            try:
                self.cursor.execute("SELECT COUNT(DISTINCT name) FROM face")
                count = self.cursor.fetchone()[0]
                self.label_cnt_face_in_database['text'] = str(count)
            except Exception as e:
                logging.error(f"查询数据库失败: {e}")
                self.label_cnt_face_in_database['text'] = "0"
        else:
            self.label_cnt_face_in_database['text'] = "0"

    def GUI_info(self):
        """初始化GUI界面"""
        # 标题
        tk.Label(self.frame_right_info,
                 text="人脸注册系统",
                 font=self.font_title).grid(row=0, column=0, columnspan=5, sticky=tk.W, padx=2, pady=20)

        # FPS显示
        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # 数据库人脸数量
        tk.Label(self.frame_right_info, text="数据库中已有的人脸: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # 当前帧人脸数量
        tk.Label(self.frame_right_info, text="当前帧中的人脸: ").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=5, sticky=tk.W, padx=5, pady=2)

        # 姓名输入
        tk.Label(self.frame_right_info, text="姓名: ").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_name.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # 按钮
        tk.Button(self.frame_right_info, text='录入', command=self.GUI_get_input_name).grid(row=5, column=2, padx=5)
        tk.Button(self.frame_right_info, text='保存当前人脸', command=self.save_current_face).grid(
            row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=10)
        tk.Button(self.frame_right_info, text='写入数据库', command=self.save_feature_to_mysql_btn).grid(
            row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=10)

        self.log_all.grid(row=8, column=0, columnspan=5, sticky=tk.W, padx=5, pady=20)

    def GUI_get_input_name(self):
        """获取用户输入的姓名并验证"""
        self.input_name_char = self.input_name.get().strip()
        if not self.input_name_char:
            self.log_all["text"] = "请输入姓名"
            self.log_all["fg"] = "red"
            return

        # 检查姓名是否已存在
        if self.input_name_char in self.registered_names:
            if not messagebox.askyesno("确认", f"姓名'{self.input_name_char}'已存在，是否覆盖？"):
                return

        self.create_face_folder()
        self.log_all["text"] = f"准备录入: {self.input_name_char}"
        self.log_all["fg"] = "green"

    def create_face_folder(self):
        """为人脸创建文件夹"""
        self.current_face_dir = os.path.join(self.path_photos_from_camera, "person_" + self.input_name_char)

        # 如果文件夹已存在，询问是否清空
        if os.path.exists(self.current_face_dir):
            if not messagebox.askyesno("确认", f"文件夹'{self.current_face_dir}'已存在，是否清空？"):
                return
            # 清空旧照片
            for f in os.listdir(self.current_face_dir):
                file_path = os.path.join(self.current_face_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(self.current_face_dir)

        self.ss_cnt = 0
        self.face_folder_created_flag = True
        self.log_all["text"] = f"使用/新建文件夹: {self.current_face_dir}"
        self.log_all["fg"] = "green"

    def save_current_face(self):
        """保存当前人脸图像"""
        if not self.face_folder_created_flag:
            self.log_all["text"] = "请先执行录入操作输入名字"
            self.log_all["fg"] = "red"
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log_all["text"] = "摄像头获取图像失败"
            self.log_all["fg"] = "red"
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(frame_rgb, 0)

        if len(faces) != 1:
            self.log_all["text"] = "请确保当前画面中只有一个人脸"
            self.log_all["fg"] = "red"
            return

        d = faces[0]
        # 扩展人脸区域
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        w_padding = w // 3
        h_padding = h // 3
        left = max(d.left() - w_padding, 0)
        top = max(d.top() - h_padding, 0)
        right = min(d.right() + w_padding, frame_rgb.shape[1])
        bottom = min(d.bottom() + h_padding, frame_rgb.shape[0])

        face_img = frame_rgb[top:bottom, left:right]
        if face_img.size == 0:
            self.log_all["text"] = "人脸区域无效，保存失败"
            self.log_all["fg"] = "red"
            return

        self.ss_cnt += 1
        img_path = os.path.join(self.current_face_dir, f"img_face_{self.ss_cnt}.jpg")
        Image.fromarray(face_img).save(img_path)
        self.log_all["text"] = f"保存成功: {img_path}"
        self.log_all["fg"] = "green"

    def return_128d_features(self, path_img):
        """从图像中提取128维特征向量"""
        try:
            img_pil = Image.open(path_img)
            img_np = np.array(img_pil)
            img_rd = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            faces = detector(img_rd, 1)

            if len(faces) != 1:
                return None

            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
            return np.array(face_descriptor)
        except Exception as e:
            logging.error(f"提取特征失败: {e}")
            return None

    def save_feature_to_mysql(self, name, feature_vec):
        """将特征向量保存到数据库"""
        if not self.conn or not self.cursor:
            self.log_all["text"] = "数据库未连接，无法写入"
            self.log_all["fg"] = "red"
            return False

        if feature_vec is None or len(feature_vec) != 128:
            self.log_all["text"] = "无效的特征向量，无法写入数据库"
            self.log_all["fg"] = "red"
            return False

        try:
            # 准备列名和占位符
            cols = ','.join([f"x{i}" for i in range(1, 129)])
            placeholders = ','.join(['%s'] * 128)

            # 检查是否已存在
            self.cursor.execute("SELECT COUNT(*) FROM face WHERE name = %s", (name,))
            exists = self.cursor.fetchone()[0] > 0

            if exists:
                # 更新现有记录
                set_clause = ','.join([f"x{i}=%s" for i in range(1, 129)])
                sql = f"UPDATE face SET {set_clause} WHERE name=%s"
                params = feature_vec.tolist() + [name]
            else:
                # 插入新记录
                sql = f"INSERT INTO face (name, {cols}) VALUES (%s, {placeholders})"
                params = [name] + feature_vec.tolist()

            self.cursor.execute(sql, params)
            self.conn.commit()

            # 更新已注册名字列表
            if name not in self.registered_names:
                self.registered_names.append(name)
                self.label_cnt_face_in_database['text'] = str(len(self.registered_names))

            return True
        except Exception as e:
            self.conn.rollback()
            self.log_all["text"] = f"数据库操作失败: {str(e)}"
            self.log_all["fg"] = "red"
            return False

    def save_feature_to_mysql_btn(self):
        """保存按钮的事件处理"""
        if not self.face_folder_created_flag:
            self.log_all["text"] = "请先录入名字并保存人脸"
            self.log_all["fg"] = "red"
            return

        if not os.path.exists(self.current_face_dir):
            self.log_all["text"] = "人脸文件夹不存在"
            self.log_all["fg"] = "red"
            return

        photos = [f for f in os.listdir(self.current_face_dir)
                  if f.startswith("img_face_") and f.endswith(".jpg")]

        if not photos:
            self.log_all["text"] = "没有人脸图像可保存"
            self.log_all["fg"] = "red"
            return

        success_count = 0
        for photo in photos:
            feature_128d = self.return_128d_features(os.path.join(self.current_face_dir, photo))
            if feature_128d is not None:
                if self.save_feature_to_mysql(self.input_name_char, feature_128d):
                    success_count += 1

        if success_count > 0:
            self.log_all["text"] = f"成功写入数据库 {success_count} 条记录: {self.input_name_char}"
            self.log_all["fg"] = "green"
        else:
            self.log_all["text"] = "未能成功写入任何记录"
            self.log_all["fg"] = "red"

    def update_fps(self):
        """更新FPS显示"""
        now = time.time()
        if str(self.start_time).split('.')[0] != str(now).split('.')[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        if self.frame_time != 0:
            self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = f"{self.fps:.2f}"

    def process(self):
        """处理摄像头帧"""
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("错误", "无法获取摄像头画面")
            self.win.quit()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(frame_rgb, 0)
        self.current_frame_faces_cnt = len(faces)
        self.label_face_cnt["text"] = str(self.current_frame_faces_cnt)

        # 绘制人脸框
        for d in faces:
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            cv2.rectangle(frame_rgb, (left, top), (right, bottom), (255, 255, 255), 2)

        self.update_fps()
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label.img_tk = img_tk
        self.label.configure(image=img_tk)

        self.win.after(20, self.process)

    def run(self):
        """运行主程序"""
        self.process()
        self.win.mainloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        face_register = Face_Register()
        face_register.run()
    except Exception as e:
        logging.error(f"程序异常: {e}")
        messagebox.showerror("错误", f"程序发生异常: {str(e)}")