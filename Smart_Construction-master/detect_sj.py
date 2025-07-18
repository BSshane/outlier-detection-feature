from copy import deepcopy
import time
import numpy as np
import sys
import cv2
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque
import json
import logging
import os
import config
import pymysql
# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 控制台输出
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, bbox, track_id, class_id):
        self.track_id = track_id
        self.state = TrackState.Tentative
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.to_xyah(bbox).reshape(-1, 1)

        # 风险监控状态
        self.in_danger_since = None
        self.no_helmet_since = None
        self.helmet_id = None  # 关联的头盔ID
        self.in_danger_reported = False  # 危险区域上报标记
        self.no_helmet_reported = False  # 未戴头盔上报标记

    def to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h
        return np.array([x, y, a, h])

    def to_tlbr(self):
        x, y, a, h = self.kf.x[:4].flatten()
        w = a * h
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2])

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed

        measurement = self.to_xyah(bbox)
        self.kf.update(measurement)

    def get_head_region(self):
        """估算人员头部区域（上1/3部分）"""
        x1, y1, x2, y2 = self.to_tlbr()
        head_height = (y2 - y1) * 0.4  # 头部约占身体高度的40%
        head_y1 = y1
        head_y2 = y1 + head_height
        return [x1, head_y1, x2, head_y2]


class HelmetTrack:
    """头盔跟踪对象"""

    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.bbox = bbox
        self.age = 1
        self.time_since_update = 0

        # 卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.to_xyah(bbox).reshape(-1, 1)

    def to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h
        return np.array([x, y, a, h])

    def to_tlbr(self):
        x, y, a, h = self.kf.x[:4].flatten()
        w = a * h
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2])

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0
        measurement = self.to_xyah(bbox)
        self.kf.update(measurement)


class Tracker:
    """多目标跟踪器 - 分别跟踪人员和头盔"""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.person_tracks = []
        self.helmet_tracks = []
        self.next_person_id = 1
        self.next_helmet_id = 1

    def update(self, person_boxes, helmet_boxes):
        """更新人员和头盔的跟踪状态"""
        # 更新人员跟踪
        if person_boxes:
            self._update_persons(person_boxes)

        # 更新头盔跟踪
        if helmet_boxes:
            self._update_helmets(helmet_boxes)

        # 关联人员和头盔
        self._match_persons_to_helmets()

        # 返回确认的人员跟踪
        return [t for t in self.person_tracks if t.state == TrackState.Confirmed]

    def _update_persons(self, person_boxes):
        """更新人员跟踪状态"""
        # 预测所有现有轨迹
        for track in self.person_tracks:
            track.predict()

        if not person_boxes:
            # 没有新检测，标记所有轨迹为未更新
            for track in self.person_tracks:
                track.time_since_update += 1
            return

        # 计算检测与轨迹的匹配
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            person_boxes, [t.to_tlbr() for t in self.person_tracks], self.iou_threshold
        )

        # 处理匹配的检测和轨迹
        for track_idx, det_idx in matches:
            self.person_tracks[track_idx].update(person_boxes[det_idx])

        # 处理未匹配的检测（创建新轨迹）
        for det_idx in unmatched_detections:
            self.person_tracks.append(Track(
                person_boxes[det_idx],
                self.next_person_id,
                0  # 0代表person类
            ))
            self.next_person_id += 1

        # 处理未匹配的轨迹
        for track_idx in unmatched_tracks:
            self.person_tracks[track_idx].time_since_update += 1

        # 移除已删除的轨迹
        self.person_tracks = [t for t in self.person_tracks
                              if t.time_since_update <= self.max_age]

    def _update_helmets(self, helmet_boxes):
        """更新头盔跟踪状态"""
        # 预测所有现有轨迹
        for track in self.helmet_tracks:
            track.predict()

        if not helmet_boxes:
            for track in self.helmet_tracks:
                track.time_since_update += 1
            return

        # 计算检测与轨迹的匹配
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            helmet_boxes, [t.to_tlbr() for t in self.helmet_tracks], self.iou_threshold
        )

        # 处理匹配的检测和轨迹
        for track_idx, det_idx in matches:
            self.helmet_tracks[track_idx].update(helmet_boxes[det_idx])

        # 处理未匹配的检测（创建新轨迹）
        for det_idx in unmatched_detections:
            self.helmet_tracks.append(HelmetTrack(
                helmet_boxes[det_idx],
                self.next_helmet_id
            ))
            self.next_helmet_id += 1

        # 处理未匹配的轨迹
        for track_idx in unmatched_tracks:
            self.helmet_tracks[track_idx].time_since_update += 1

        # 移除已删除的轨迹
        self.helmet_tracks = [t for t in self.helmet_tracks
                              if t.time_since_update <= self.max_age]

    def _match_detections_to_tracks(self, detections, tracks, iou_threshold):
        """计算检测与轨迹的匹配"""
        if not tracks:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._iou(det, track)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = []
        unmatched_detections = []
        unmatched_tracks = list(range(len(tracks)))

        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] >= iou_threshold:
                matches.append((t, d))
                unmatched_tracks.remove(t)
            else:
                unmatched_detections.append(d)

        for d in range(len(detections)):
            if d not in row_ind:
                unmatched_detections.append(d)

        return matches, unmatched_detections, unmatched_tracks

    def _iou(self, bbox1, bbox2):
        """计算IoU"""
        x11, y11, x12, y12 = bbox1
        x21, y21, x22, y22 = bbox2

        xA = max(x11, x21)
        yA = max(y11, y21)
        xB = min(x12, x22)
        yB = min(y12, y22)

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        box1_area = (x12 - x11) * (y12 - y11)
        box2_area = (x22 - x21) * (y22 - y21)

        iou = inter_area / float(box1_area + box2_area - inter_area + 1e-10)
        return iou

    def _match_persons_to_helmets(self):
        """将人员与头盔进行关联"""
        confirmed_persons = [t for t in self.person_tracks if t.state == TrackState.Confirmed]
        active_helmets = [h for h in self.helmet_tracks if h.time_since_update < 2]

        if not confirmed_persons or not active_helmets:
            # 没有人员或头盔，清空所有关联
            for person in confirmed_persons:
                person.helmet_id = None
            return

        # 构建人员头部区域与头盔的距离矩阵
        cost_matrix = np.zeros((len(confirmed_persons), len(active_helmets)))
        for i, person in enumerate(confirmed_persons):
            person_head = person.get_head_region()
            person_head_center = ((person_head[0] + person_head[2]) / 2,
                                  (person_head[1] + person_head[3]) / 2)

            for j, helmet in enumerate(active_helmets):
                helmet_box = helmet.to_tlbr()
                helmet_center = ((helmet_box[0] + helmet_box[2]) / 2,
                                 (helmet_box[1] + helmet_box[3]) / 2)

                # 计算头部中心与头盔中心的距离
                distance = np.sqrt(
                    (person_head_center[0] - helmet_center[0]) ** 2 +
                    (person_head_center[1] - helmet_center[1]) ** 2
                )

                # 归一化距离（除以图像宽度）
                cost_matrix[i, j] = distance / 800  # 假设图像宽度为800像素

        # 使用匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 更新关联关系
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.1:  # 距离阈值（可调整）
                confirmed_persons[i].helmet_id = active_helmets[j].track_id
            else:
                confirmed_persons[i].helmet_id = None

        # 未匹配的人员设为无头盔
        for i in range(len(confirmed_persons)):
            if i not in row_ind:
                confirmed_persons[i].helmet_id = None


class YOLOPredict(object):
    def __init__(self, weights, out_file_path, risk_queue=None, rtmp_url="camera_1"):
        """
        YOLO 模型初始化
        :param weights: 权重路径
        :param out_file_path: 推理结果存放路径
        :param risk_queue: 风险信息队列
        :param rtmp_url: RTSP流地址（用于获取cameraId）
        """

        '''模型参数'''
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        self.conf_thres = 0.4
        self.device = ''
        self.img_size = 640
        self.iou_thres = 0.5
        self.output = out_file_path
        self.save_txt = False
        self.update = False
        self.view_img = False
        self.weights = weights  # 权重文件路径
        self.rtmp_url = rtmp_url  # 新增：RTSP流地址
        # 数据库配置
        self.db_config = config.db_config or {
            'host': 'localhost',
            'user': 'your_username',
            'password': 'your_password',
            'database': 'your_database',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

        # 初始化数据库连接
        self._init_db_connection()
        # 初始化目标跟踪器
        self.tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)

        # 加载模型
        self.model, self.half, self.names, self.colors, self.device = self.load_model()

        self.predict_info = ""

        # 创建视频帧队列
        self.frame_pool = deque(maxlen=150)

        # 风险信息队列
        self.risk_queue = risk_queue

    def _init_db_connection(self):
        """初始化数据库连接"""
        try:
            self.db_conn = pymysql.connect(**self.db_config)
            logger.info("数据库连接已建立")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            self.db_conn = None

    def _ensure_db_connection(self):
        """确保数据库连接有效，必要时重新连接"""
        if not self.db_conn or not self.db_conn.open:
            logger.warning("数据库连接丢失，尝试重新连接")
            try:
                self.db_conn = pymysql.connect(**self.db_config)
                return True
            except Exception as e:
                logger.error(f"重新连接数据库失败: {e}")
                return False
        return True

    def risk_report(self, message, person_id):
        """风险上报函数 - 直接写入数据库"""
        # 过滤非目标类型的告警
        if not any(keyword in message.lower() for keyword in ['helmet', 'danger', 'zone']):
            return  # 忽略其他类型的告警

        # 获取当前时间戳
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        # 从message中解析风险类型
        risk_type = self._parse_risk_type(message)
        if risk_type not in ['helmet', 'dangerous area']:
            return  # 忽略其他类型的告警

        # 生成视频URL
        video_url = self._generate_video_url(current_time,risk_type,person_id)
        camera_id_str = self._get_camera_id()
        camera_id = int(camera_id_str.split('_')[-1])  # 提取数字部分
        # 构建完整的告警信息
        warning_info = {
            "warning": message,
            "cameraId": camera_id,
            "type": risk_type,
            "videoURL": video_url,
            "info": self._generate_detail_info(message, risk_type)
        }

        # 保存风险相关的视频或图片
        self.save_risk_media(risk_type, person_id, current_time)

        # 保留原有打印功能
        print(f"【风险上报】{json.dumps(warning_info, ensure_ascii=False, indent=2)}")

        # 直接写入数据库
        self._insert_risk_to_db(warning_info)

    def _insert_risk_to_db(self, warning_info):
        """将风险信息写入数据库"""
        if not self._ensure_db_connection():
            logger.error("无法连接数据库，风险信息写入失败")
            return False

        try:
            with self.db_conn.cursor() as cursor:
                # 使用你提供的SQL语句
                sql = """
                      INSERT INTO warning (cameraId, type, videoURL, info)
                      VALUES (%s, %s, %s, %s) \
                      """

                # 执行插入
                cursor.execute(sql, (
                    warning_info['cameraId'],
                    warning_info['type'],
                    warning_info['videoURL'],
                    json.dumps(warning_info['info'])
                ))

            # 提交事务
            self.db_conn.commit()
            logger.info(f"风险信息已写入数据库: {warning_info.get('warning', '未知风险')}")
            return True

        except Exception as e:
            logger.error(f"写入数据库失败: {e}")
            # 回滚事务
            self.db_conn.rollback()
            return False

    def _parse_risk_type(self, message):
        """从消息中解析风险类型（仅支持helmet和dangerous area）"""
        message_lower = message.lower()
        if 'helmet' in message_lower or '头盔' in message:
            return 'helmet'
        elif 'danger' in message_lower or 'zone' in message_lower or '危险区域' in message:
            return 'dangerous area'
        return 'unknown'  # 不会被使用，因为外层已过滤

    def _generate_video_url(self, timestamp, risk_type, person_id):
        """生成告警视频链接"""
        # 规范化时间戳，确保不包含路径非法字符
        safe_timestamp = timestamp.replace(':', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')
        # 根据风险类型确定文件后缀
        if risk_type == 'helmet':
            file_extension = 'jpg'
        elif risk_type == 'dangerous area':
            file_extension = 'mp4'
        else:
            file_extension = 'mp4'

        media_name = f"{risk_type.replace(' ', '_')}-{person_id}-{safe_timestamp}.{file_extension}"
        # 返回完整的媒体链接
        return f"http://127.0.0.1:9081/{media_name}"

    def _get_camera_id(self):
        """获取当前摄像头ID（从配置或连接信息中提取）"""
        # 从rtmp_url中提取cameraId
        return self.rtmp_url.split('/')[-1]

    def _generate_detail_info(self, message, risk_type):
        """生成详细告警信息（根据风险类型定制）"""
        if risk_type == 'helmet':
            return {
                "risk_level": "medium",
                "location": "Work Area",
                "description": f"人员未佩戴头盔: {message}"
            }
        elif risk_type == 'dangerous area':
            return {
                "risk_level": "high",
                "location": "Danger Zone A",
                "description": f"人员进入危险区域: {message}"
            }
        return {"description": message}  # 默认情况

    def is_point_in_danger_zone(self, x, y, danger_zone):
        """判断点(x,y)是否在危险区域内"""
        x1, y1, x2, y2 = danger_zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_helmet_on_person(self, person_box, helmet_boxes):
        """判断人员是否戴头盔（通过边界框重叠判断）"""
        px1, py1, px2, py2 = person_box
        for (hx1, hy1, hx2, hy2) in helmet_boxes:
            # 计算两个矩形的重叠面积
            overlap_x1 = max(px1, hx1)
            overlap_y1 = max(py1, hy1)
            overlap_x2 = min(px2, hx2)
            overlap_y2 = min(py2, hy2)
            overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
            # 若重叠面积超过人员头部区域的1/3，则认为戴了头盔
            person_head_area = (px2 - px1) * (py2 - py1) * 0.3  # 头部约占身体30%
            if overlap_area > person_head_area:
                return True
        return False

    def load_model(self):
        """加载模型"""
        imgsz = self.img_size
        weights = self.weights
        device = torch_utils.select_device(device=self.device)
        print(f"当前使用设备: {device}")
        half = device.type != 'cpu'  # 仅GPU支持半精度

        # 加载模型
        model = attempt_load(weights, map_location=device)
        imgsz = check_img_size(imgsz, s=model.stride.max())
        if half:
            model.half()  # 转换为半精度

        # 获取类别名称和颜色
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # 初始化模型
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        return model, half, names, colors, device

    def detect(self, source, danger_zone=(100, 100, 700, 700), helmet_timeout=5, danger_timeout=10, save_img=False):
        """
        推理主函数 - 优化危险区域检测及风险上报逻辑
        :param source: 推理素材（视频帧或地址）
        :param danger_zone: 危险区域 (x1,y1,x2,y2)
        :param helmet_timeout: 未戴头盔报警时长（秒）
        :param danger_timeout: 危险区域报警时长（秒）
        :param save_img: 是否保存图片
        """
        # 危险区域坐标校验与修正
        x1, y1, x2, y2 = danger_zone
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        danger_zone = (x1, y1, x2, y2)
        print(f"有效危险区域: {danger_zone} (x1<=x2, y1<=y2)")

        out = self.output
        view_img = self.view_img
        save_txt = self.save_txt
        imgsz = self.img_size
        augment = self.augment
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        cclasses = self.classes
        agnostic_nms = self.agnostic_nms
        update = self.update

        os.makedirs(out, exist_ok=True)
        t0 = time.time()

        # 判断输入类型（单帧或视频源）
        is_single_frame = isinstance(source, np.ndarray)
        if is_single_frame:
            webcam = False
            img = source
            im0s = source.copy()
            dataset = [('frame', img, im0s, None, 'Processing frame')]
        else:
            webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
            if webcam:
                view_img = True
                cudnn.benchmark = True
                dataset = LoadStreams(source, img_size=imgsz)
            else:
                save_img = True
                dataset = LoadImages(source, img_size=imgsz, visualize_flag=True)

        result_frame = None
        vid_path, vid_writer = None, None

        for path, img, im0s, vid_cap, info_str in dataset:
            # 预处理图像
            if isinstance(img, np.ndarray):
                img = letterbox(img, new_shape=imgsz)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
            else:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

            # 模型推理
            t1 = torch_utils.time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=cclasses, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # 处理检测结果
            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                # 汇总当前帧的风险信息
                frame_risk_info = []

                # 1. 绘制危险区域（无论是否有检测目标）
                cv2.rectangle(im0, (danger_zone[0], danger_zone[1]),
                              (danger_zone[2], danger_zone[3]),
                              (0, 0, 255), 2)  # 红色框

                # 2. 绘制危险区域标签和Risk Summary标签（紧挨着）
                danger_label = "Danger Zone"
                risk_label = "Risk Summary:"
                (dw, dh), _ = cv2.getTextSize(danger_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                (rw, rh), _ = cv2.getTextSize(risk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # 计算两个标签的位置（紧挨着）
                x_pos = danger_zone[0]
                y_pos = max(10, danger_zone[1] - 10)  # 确保不超出图像

                cv2.putText(im0, danger_label, (x_pos, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 第二个标签紧挨着第一个，中间留5像素空格
                cv2.putText(im0, risk_label, (x_pos + dw + 5, y_pos + 3),  # +3微调垂直对齐
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if det is not None and len(det):
                    # 调整边界框坐标
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # 提取人员和头盔的边界框
                    persons = []  # 人员边界框 (x1,y1,x2,y2)
                    helmets = []  # 头盔边界框 (x1,y1,x2,y2)
                    for *xyxy, conf, cls in det:
                        class_name = self.names[int(cls)]
                        # 根据类别设置不同颜色
                        if class_name == 'person':
                            color = (255, 0, 0)  # 蓝色
                        elif class_name == 'head':
                            color = (0, 0, 0)
                        elif class_name == 'helmet':
                            color = (255, 255, 255)  # 白色
                        else:
                            color = self.colors[int(cls)]

                        # 绘制所有目标的检测框
                        label = f'{class_name} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=2)
                        # 分类存储边界框
                        if class_name == 'person':
                            persons.append((xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))
                        elif class_name == 'helmet':
                            helmets.append((xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))

                    # 更新跟踪器
                    tracked_persons = self.tracker.update(persons, helmets)
                    print(f"当前帧检测到 {len(tracked_persons)} 个确认的人员")

                    # 风险检测逻辑
                    current_time = time.time()
                    for track in tracked_persons:
                        x1, y1, x2, y2 = track.to_tlbr()
                        track_id = track.track_id
                        person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        person_box = (x1, y1, x2, y2)

                        # 显示人员ID和位置信息
                        debug_info = f"ID:{track_id}"
                        cv2.putText(im0, debug_info, (int(x1), max(10, int(y1) - 50)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # 1. 危险区域判断
                        in_danger = self.is_point_in_danger_zone(
                            person_center[0], person_center[1], danger_zone
                        )

                        # 危险区域状态管理
                        if in_danger:
                            if track.in_danger_since is None:
                                track.in_danger_since = current_time
                                track.in_danger_reported = False
                                frame_risk_info.append(f"Person {track_id} entered danger zone")

                            # 计算停留时间
                            duration = current_time - track.in_danger_since

                            # 2. 改变危险区域内人员框的样式（粗线+红色）
                            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)),
                                          (0, 0, 255), 4)  # 红色粗框

                            # 显示停留时间
                            cv2.putText(im0, f"Danger: {duration:.1f}s",
                                        (int(x1), max(10, int(y1) - 30)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # 超时判断与上报
                            if duration > danger_timeout and not track.in_danger_reported:
                                risk_msg = f"Person {track_id} overstay in danger zone: {duration:.1f}s"
                                self.risk_report(risk_msg, track_id)
                                frame_risk_info.append(risk_msg)
                                track.in_danger_reported = True
                        else:
                            # 离开危险区域：重置状态
                            if track.in_danger_since is not None:
                                track.in_danger_since = None
                                track.in_danger_reported = False

                        # 3. 头盔佩戴判断
                        has_helmet = self.is_helmet_on_person(person_box, helmets)

                        if not has_helmet:
                            if track.no_helmet_since is None:
                                track.no_helmet_since = current_time
                                track.no_helmet_reported = False
                                frame_risk_info.append(f"Person {track_id} not wearing helmet")

                            duration = current_time - track.no_helmet_since
                            cv2.putText(im0, f"No helmet: {duration:.1f}s",
                                        (int(x1), int(y2) + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            if duration > helmet_timeout and not track.no_helmet_reported:
                                risk_msg = f"Person {track_id} no helmet over {duration:.1f}s"
                                self.risk_report(risk_msg, track_id)
                                frame_risk_info.append(risk_msg)
                                track.no_helmet_reported = True
                        else:
                            if track.no_helmet_since is not None:
                                track.no_helmet_since = None
                            cv2.putText(im0, "Helmet on",
                                        (int(x1), int(y2) + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 4. 组合风险判断（同时在危险区域且未戴头盔）
                        if in_danger and not has_helmet:
                            combined_duration = current_time - max(
                                track.in_danger_since or current_time,
                                track.no_helmet_since or current_time
                            )
                            cv2.putText(im0, f"HIGH RISK!",
                                        (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if combined_duration > max(helmet_timeout, danger_timeout):
                                frame_risk_info.append(f"Person {track_id} HIGH RISK: in danger zone without helmet")

                # 5. 在图像上显示当前帧的风险汇总信息
                if frame_risk_info:
                    # 逐条显示风险信息（从Risk Summary标签下方开始）
                    for idx, info in enumerate(frame_risk_info[:5]):  # 最多显示5条
                        y_pos = max(10, danger_zone[1] + 15 + idx * 25)  # 从危险区域上方开始
                        cv2.putText(im0, info, (danger_zone[0], y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 无风险时显示正常状态
                    y_pos = max(10, danger_zone[1] + 15)
                    cv2.putText(im0, "No risks detected", (danger_zone[0], y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 打印推理时间
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # 将结果帧入队
                self.frame_pool.append(im0)

                # 保存结果
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else 640
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else 480
                            fourcc = 'mp4v'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        if vid_writer:
                            vid_writer.write(im0)

        # 保存当前帧（单帧输入时）
        if isinstance(source, np.ndarray):
            result_frame = im0

        # 释放资源
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()

        if save_txt or save_img:
            print('结果保存至 %s' % str(out))
        print('总耗时: (%.3fs)' % (time.time() - t0))

        return result_frame if isinstance(source, np.ndarray) else save_path

    def save_risk_media(self, risk_type, person_id, timestamp):
        """保存风险相关的视频或图片"""
        import ntpath

        # 使用项目相对路径或配置路径（避免使用绝对路径根目录）
        base_dir = os.path.join(os.getcwd(), "FileGetter")

        # 确保目录存在，添加错误处理
        try:
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"资源目录检查: {base_dir} 存在")
        except Exception as e:
            logger.error(f"创建资源目录失败: {e}")
            return

        # 规范化时间戳，确保不包含路径非法字符
        safe_timestamp = timestamp.replace(':', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')

        try:
            if risk_type == 'dangerous area':
                # 保存视频文件
                video_name = f"{risk_type.replace(' ', '_')}-{person_id}-{safe_timestamp}.mp4"
                video_path = os.path.join(base_dir, video_name)

                # 检查帧队列是否有足够的帧
                if len(self.frame_pool) < 10:
                    logger.warning(f"帧队列帧数量不足({len(self.frame_pool)}/150)，使用现有帧保存视频")

                # 获取第一帧尺寸作为视频尺寸
                frame = self.frame_pool[0] if self.frame_pool else np.zeros((480, 640, 3), dtype=np.uint8)
                h, w = frame.shape[:2]

                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

                # 写入帧并添加进度日志
                frame_count = min(150, len(self.frame_pool))
                for i, frame in enumerate(list(self.frame_pool)[-frame_count:]):
                    out.write(frame)
                    if i % 30 == 0:
                        logger.debug(f"正在写入视频帧 {i + 1}/{frame_count}")

                out.release()
                logger.info(f"危险区域风险视频已保存: {video_path}")

            elif risk_type == 'helmet':
                # 保存图片
                image_name = f"{risk_type}-{person_id}-{safe_timestamp}.jpg"
                image_path = os.path.join(base_dir, image_name)

                # 检查帧队列是否有内容
                if not self.frame_pool:
                    logger.warning("帧队列为空，无法保存图片")
                    return

                # 尝试保存图片，添加错误处理
                success = cv2.imwrite(image_path, self.frame_pool[-1])

                if success:
                    logger.info(f"未戴头盔风险图片已保存: {image_path}")

                    # 验证文件是否实际存在（Windows特性检查）
                    if os.path.exists(image_path):
                        file_size = os.path.getsize(image_path)
                        logger.info(f"图片验证成功，大小: {file_size} 字节")
                    else:
                        logger.warning(f"图片保存后未找到文件，可能被系统拦截")
                else:
                    logger.error(f"保存图片失败: {image_path}")

                    # 尝试使用PNG格式保存（兼容性更好）
                    image_path_png = os.path.splitext(image_path)[0] + ".png"
                    success_png = cv2.imwrite(image_path_png, self.frame_pool[-1])

                    if success_png:
                        logger.info(f"未戴头盔风险图片已保存为PNG: {image_path_png}")
                    else:
                        # 尝试保存为BMP（最基本的图像格式）
                        image_path_bmp = os.path.splitext(image_path)[0] + ".bmp"
                        success_bmp = cv2.imwrite(image_path_bmp, self.frame_pool[-1])

                        if success_bmp:
                            logger.info(f"未戴头盔风险图片已保存为BMP: {image_path_bmp}")
                        else:
                            # 输出帧数据调试信息
                            last_frame = self.frame_pool[-1]
                            logger.error(f"所有图片保存尝试均失败，帧数据检查: "
                                         f"形状={last_frame.shape}, 类型={last_frame.dtype}, "
                                         f"像素范围=[{last_frame.min()}, {last_frame.max()}]")

        except Exception as e:
            # 捕获并记录所有异常
            logger.error(f"保存风险媒体文件时发生未知错误: {e}", exc_info=True)

if __name__ == '__main__':
    print("请通过主程序调用 YOLOPredict 类")