import os
import time
import json
import logging
import signal
import cv2
import numpy as np
import pymysql
from flask import Flask, Response, request, jsonify, render_template
from flasgger import Swagger
from flask_cors import CORS
from multiprocessing import Process, active_children
from pathlib import Path
from detect_sj import YOLOPredict
from config import (
    rtmp_url, get_weights_path, db_config,
    current_danger_zone, risk_queue, stop_event
)
from processes import reader_process  # 直接导入读者进程
from config import risk_queue
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# python ./Smart_Construction-master/app.py
# 初始化Flask应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})
swagger = Swagger(app)

# 初始化模型
weights_path = get_weights_path()
logger.info(f"将使用权重文件: {weights_path}")
model = YOLOPredict(weights_path, "inference/output")

# 风险信息池
risk_pool = []

# 保存危险区域到数据库
def save_danger_zone_to_db(x1, y1, x2, y2):
    try:
        with pymysql.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                camera_id = rtmp_url.split('/')[-1]
                insert_query = """
                               INSERT INTO danger (cameraId, x1, y1, x2, y2)
                               VALUES (%s, %s, %s, %s, %s) \
                               """
                cursor.execute(insert_query, (camera_id, x1, y1, x2, y2))
                conn.commit()
        logger.info(f"危险区域已保存到数据库: {camera_id} - ({x1}, {y1}, {x2}, {y2})")
        return True
    except Exception as e:
        logger.error(f"保存危险区域时发生数据库错误: {e}")
        return False

# 视频帧生成器
def generate_frames():
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        logger.error(f"无法打开视频流: {rtmp_url}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return

    logger.info(f"成功打开视频流: {rtmp_url}")
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            logger.warning("无法读取视频帧，尝试重新连接...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtmp_url)
            continue

        # 处理视频帧（检测目标并判断是否进入危险区域）
        try:
            processed_frame = model.detect(frame, danger_zone=current_danger_zone)
            # 确保detect方法将风险信息放入队列
            # 这部分逻辑需要在detect_sj.py中正确实现
        except Exception as e:
            logger.error(f"处理视频帧时出错: {e}")
            processed_frame = frame

        # 编码为JPEG并返回
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            logger.warning("无法编码视频帧")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)  # 控制帧率

    cap.release()
    logger.info("视频流已关闭")

# 信号处理：优雅关闭
def handle_shutdown(signum, frame):
    logger.info(f"接收到信号 {signum}，开始优雅关闭...")
    stop_event.set()

    # 终止所有子进程
    for proc in active_children():
        logger.info(f"终止子进程: {proc.name} (PID: {proc.pid})")
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            logger.warning(f"进程 {proc.name} 未能及时结束")

    logger.info("应用已优雅关闭")
    os._exit(0)

# 注册信号处理
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# API路由
@app.route('/video_feed')
def video_feed():
    """视频流接口"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_danger_zone', methods=['POST'])
def set_danger_zone():
    """设置危险区域接口"""
    global current_danger_zone
    data = request.get_json()
    x1, y1, x2, y2 = data.get('x1'), data.get('y1'), data.get('x2'), data.get('y2')

    # 验证输入
    if not all([isinstance(x1, int), isinstance(y1, int),
                isinstance(x2, int), isinstance(y2, int)]):
        return jsonify({'message': '坐标必须为整数'}), 400

    # 调整坐标顺序（确保x1<x2, y1<y2）
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    current_danger_zone = (x1, y1, x2, y2)
    saved = save_danger_zone_to_db(x1, y1, x2, y2)
    if saved:
        return jsonify({'message': '危险区域已更新并保存到数据库'})
    return jsonify({'message': '危险区域已更新，但保存到数据库时出错'}), 500

@app.route('/get_risk_pool')
def get_risk_pool():
    """获取风险池信息接口"""
    return jsonify(risk_pool)

@app.route('/get_process_status')
def get_process_status():
    """获取进程状态接口"""
    try:
        queue_size = risk_queue.qsize()
    except Exception:
        queue_size = -1

    active_processes = [{
        'name': p.name,
        'pid': p.pid,
        'is_alive': p.is_alive()
    } for p in active_children()]

    reader_alive = any(p.name == "RiskReaderProcess" and p.is_alive()
                       for p in active_children())

    return jsonify({
        'reader_process': reader_alive,
        'queue_size': queue_size,
        'active_processes': active_processes
    })

@app.route('/')
def index():
    """主页接口"""
    return render_template('index.html')

# app.py
# ...已有代码...

if __name__ == '__main__':
    reader = None  # 初始化读者进程变量

    try:
        # 直接启动读者进程（不作为守护进程）
        reader = Process(
            target=reader_process,
            args=(risk_pool,),
            name="RiskReaderProcess"
        )
        reader.start()
        if reader.is_alive():
            logger.info(f"读者进程已成功启动，PID: {reader.pid}")
        else:
            logger.error("读者进程启动失败")

        # 启动Flask应用
        logger.info("应用启动，等待连接...")
        app.run(debug=False, host='0.0.0.0', port=9086, use_reloader=False)

    except Exception as e:
        logger.critical(f"主程序启动失败: {e}", exc_info=True)
        stop_event.set()
    finally:
        # 确保所有进程都被正确终止
        stop_event.set()

        if reader and reader.is_alive():
            logger.info("正在停止读者进程...")
            reader.terminate()
            reader.join(timeout=5)
            if reader.is_alive():
                logger.warning("读者进程未能及时停止")

        logger.info("主程序已退出")

