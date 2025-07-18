from flask import Flask, jsonify, request, send_file,Response
import yaml
import os
from pathlib import Path
import time
from flask_cors import CORS
import threading # 引入线程，用于管理帧和状态
import io # 用于处理字节流
import cv2 # 用于处理图像

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# 全局变量，用于存储最新接收到的帧的字节数据
latest_frame_bytes = None
# 用于保护 latest_frame_bytes 的锁
frame_lock = threading.Lock()
# 用于指示 detect.py 是否正在发送帧
detection_active = False
# 记录最后一次收到帧的时间
last_frame_time = 0

# 加载YAML配置
try:
    with open("frontend_config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: frontend_config.yml not found. Using default configuration.")
    config = {
        "界面": {
            "标题": "默认跌倒检测系统",
            "组件": [
                {"类型": "图片显示", "路径": "runs/detect", "刷新频率": 1500, "最大宽度": 800},
                {"类型": "状态文本", "内容": "系统正在初始化..."}
            ]
        }
    }

# 从配置中获取参数
# 注意：这里我们不再需要 base_image_output_dir 和 find_latest_exp_dir，
# 因为图片是直接通过内存传输的，不再依赖本地文件系统
refresh_rate = config["界面"]["组件"][0]["刷新频率"]
max_width = config["界面"]["组件"][0]["最大宽度"]
title = config["界面"]["标题"]
status_text_initial = config["界面"]["组件"][1]["内容"] # 初始状态文本

# 后台线程，用于更新 detection_active 状态
def update_detection_status():
    global detection_active
    global last_frame_time
    while True:
        # 如果在过去2秒内没有收到帧，则认为 detect.py 不活跃
        if time.time() - last_frame_time > 2:
            if detection_active:
                print("Detect process is inactive (no frames received recently).")
            detection_active = False
        else:
            if not detection_active:
                print("Detect process is active and sending frames.")
            detection_active = True
        time.sleep(1) # 每秒检查一次

status_thread = threading.Thread(target=update_detection_status, daemon=True)
status_thread.start()

@app.route('/api/config', methods=['GET'])
def get_config():
    current_status_text = status_text_initial
    if not detection_active:
        current_status_text = "等待检测服务启动并发送帧..."
    else:
        current_status_text = "检测服务运行中，正在接收帧..."

    # return jsonify({
    #     'title': title,
    #     'refresh_rate': refresh_rate, # 刷新频率传递给前端
    #     'max_width': max_width,
    #     'status_text': current_status_text
    # })
    return jsonify({'title': title, 'status_text': current_status_text})

# 新增：接收 detect.py 上传帧的路由
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame_bytes
    global last_frame_time
    if request.headers['Content-Type'] == 'image/jpeg':
        with frame_lock:
            latest_frame_bytes = request.data
            last_frame_time = time.time() # 更新最后收到帧的时间
        return 'Frame received', 200
    return 'Invalid content type', 400

def generate_frames():
    """一个生成器函数，持续地从内存中读取最新帧并作为流的一部分发送。"""
    while True:
        with frame_lock:
            if latest_frame_bytes is None:
                # 如果还没有帧，可以等待一小会
                time.sleep(0.01)
                continue
            frame_to_send = latest_frame_bytes

        # 使用 multipart/x-mixed-replace 格式产生流
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        # 控制一下发送速率，避免给浏览器太大压力
        # 这个值可以根据实际情况调整，例如 0.033 约等于 30fps
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    """这个路由返回一个流式响应，前端可以直接用在 <img> 标签里。"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# # 修改：获取最新图片的路由，直接从内存中返回图片
# @app.route('/get_latest_image', methods=['GET'])
# def get_latest_image():
#     with frame_lock:
#         if latest_frame_bytes:
#             # 使用 io.BytesIO 将字节数据包装成文件对象，然后使用 send_file 发送
#             return send_file(io.BytesIO(latest_frame_bytes), mimetype='image/jpeg')
#     # 如果没有可用的图片，返回一个空的响应或者一个默认图片
#     return "No image available", 204 # 204 No Content
#
# @app.route('/')
# def home():
#     return "Flask Backend for Fall Detection System is running."

# --- 为了方便测试，提供一个简单的HTML页面来直接显示视频流 ---
@app.route('/')
def index():
    """主页，直接显示视频流，无需复杂的前端框架。"""
    html = f"""
    <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: sans-serif; text-align: center; background-color: #f0f0f0; }}
                h1 {{ color: #333; }}
                img {{ border: 2px solid #ccc; border-radius: 8px; margin-top: 20px; max-width: 90%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p id="status">{status_text_initial}</p>
            <img src="/video_feed" width="800">

            <script>
                // 使用JS轮询/api/status来更新状态文本
                function fetchStatus() {{
                    fetch('/api/config')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('status').innerText = data.status_text;
                            document.title = data.title; // 也可以更新页面标题
                        }})
                        .catch(error => console.error('Error fetching status:', error));
                }}
                // 页面加载时先调用一次，然后每2秒更新一次
                fetchStatus();
                setInterval(fetchStatus, 2000);
            </script>
        </body>
    </html>
    """
    return html

if __name__ == '__main__':
    # 移除 find_latest_exp_dir 和 app.static_folder 的设置，因为不再依赖本地文件
    app.run(debug=True, host='0.0.0.0', port=5000)