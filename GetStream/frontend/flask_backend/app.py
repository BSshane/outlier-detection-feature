from flask import Flask, jsonify, request
from flask_cors import CORS
from flasgger import Swagger, swag_from
import re
import os
import subprocess
import uuid


config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/apidocs/" # Swagger UI 的访问路径
}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 限制跨域请求到 /api/* 路径
swagger = Swagger(app, config=config)

PATH_TO_FACE_RECOG_INTERPRETER=r"C:\Users\renyi\miniconda3\envs\GiteeADDlib\python.exe" #人脸识别虚拟环境路径
PATH_TO_PROCESSOR_SCRIPT=r"D:\1 MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\GetStream\frontend\flask_backend\SP_VideoProcess.py"
TEMP_IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'static', 'captures')# 存放临时路径
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)




@app.route('/api/video', methods=['GET', 'POST'])
def get_video():
    """
    视频流地址接口
    这个接口用于处理视频流地址。
    - GET: 获取一个默认的HLS视频流地址。
    - POST: 接收一个RTMP推流地址，并将其转换为对应的HLS拉流地址。
    ---
    get:
      tags:
        - Video Stream
    summary: 获取默认的HLS视频流地址
    description: 用于测试或演示，直接返回一个硬编码的HLS视频流URL。
    responses:
      200:
        description: 成功返回视频流地址
        schema:
          type: object
          properties:
             video_url:
               type: string
               example: "http://1.92.135.70/hls/camera1.m3u8"

    post:
      tags:
        - Video Stream
      summary: 转换RTMP地址为HLS地址
      description: 接收包含RTMP地址的JSON请求，解析后返回对应的HLS地址。
      parameters:
        - name: body
          in: body
          required: true
          schema:
            id: RtmpInput
            type: object
            required:
              - rtmp_url
            properties:
              rtmp_url:
                type: string
                description: "要转换的RTMP推流地址"
                example: "rtmp://your-server-ip/live/streamkey"
    responses:
      200:
        description: 成功转换并返回HLS地址
        schema:
          type: object
          properties:
            video_url:
              type: string
              description: "转换后的HLS拉流地址"
              example: "http://1.92.135.70/hls/streamkey.m3u8"
      400:
        description: "请求错误，例如URL为空或格式不正确"
        schema:
          type: object
          properties:
            error:
              type: string
              example: "请输入有效的 RTMP URL（必须以 rtmp:// 开头）"
      500:
        description: "服务器内部错误，例如URL解析失败"
        schema:
          type: object
          properties:
            error:
              type: string
              example: "RTMP URL 解析失败: [具体错误信息]"
        """
    if request.method == 'POST':
        data = request.get_json()
        rtmp_url = data.get('rtmp_url', '').strip()  # 去除首尾空格
        # 验证 RTMP URL 格式
        if not rtmp_url:
            return jsonify({"error": "RTMP URL 不能为空"}), 400
        if not rtmp_url.startswith('rtmp://'):
            return jsonify({"error": "请输入有效的 RTMP URL（必须以 rtmp:// 开头）"}), 400
        # 使用正则表达式提取流名称 (stream key)
        try:
            match = re.search(r'rtmp://[^/]+/[^/]+/([^/]+)', rtmp_url)
            if not match:
                return jsonify({"error": "RTMP URL 格式不正确，无法解析流名称"}), 400
            stream_key = match.group(1)
            # 转换为 HLS URL，假设服务器支持 /hls/<stream_key>.m3u8
            hls_url = f"http://1.92.135.70/hls/{stream_key}.m3u8"
            return jsonify({"video_url": hls_url})
        except Exception as e:
            return jsonify({"error": f"RTMP URL 解析失败: {str(e)}"}), 500
    # 默认 GET 请求返回示例 HLS URL
    return jsonify({"video_url": "http://1.92.135.70/hls/camera1.m3u8"})


@app.route('/api/capture-frame', methods=['POST'])
def capture_frame_endpoint():
    """
    从RTMP流截取一帧图像
    接收一个RTMP地址，调用外部Python脚本进行截图，并直接返回图片文件。
    ---
    post:
      tags:
        - Video Capture
      summary: 截取视频流的当前帧
      description: 接收一个RTMP推流地址，调用后台进程截取一帧画面并返回图片。
      parameters:
        - name: body
          in: body
          required: true
          schema:
            type: object
            required:
              - rtmp_url
            properties:
              rtmp_url:
                type: string
                description: "要从中截图的RTMP推流地址"
                example: "rtmp://your-server-ip/live/streamkey"
      consumes:
        - application/json
      produces:
        - image/jpeg  # 明确指出成功时返回的是图片
      responses:
        200:
          description: 成功返回截取的图片文件。
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        400:
          description: "请求错误，例如rtmp_url为空。"
        500:
          description: "服务器内部错误，例如脚本执行失败或路径配置错误。"
    """
    data = request.get_json()
    if not data or 'rtmp_url' not in data:
        return jsonify({"error": "Request body must be JSON with 'rtmp_url' key"}), 400

    rtmp_url = data['rtmp_url'].strip()
    if not rtmp_url:
        return jsonify({"error": "rtmp_url cannot be empty"}), 400

    # 1. 为输出图片生成一个唯一的、安全的文件路径
    unique_filename = f"{uuid.uuid4()}.jpg"
    output_image_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)

    # 2. 构建将要执行的命令列表
    command = [
        PATH_TO_FACE_RECOG_INTERPRETER,
        PATH_TO_PROCESSOR_SCRIPT,
        '--rtmp-url', rtmp_url,
        '--output-path', output_image_path
    ]

    # 3. --- 关键的subprocess调用逻辑 ---
    try:
        # 执行命令，并设置超时（例如30秒）
        # check=True 表示如果脚本返回非0退出码（即出错），会抛出异常
        # capture_output=True 会捕获脚本的打印输出，方便调试
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        # 如果脚本成功执行，result.stdout会包含脚本的打印信息
        print("Subprocess execution successful, stdout:", result.stdout)

        # 4. 成功后，将生成的图片文件发送回前端
        return send_file(output_image_path, mimetype='image/jpeg')

    except FileNotFoundError:
        # 捕获因路径错误导致找不到解释器或脚本的错误
        print(f"Error: Could not find Python interpreter or script. Check paths.")
        return jsonify({
            "error": "Server configuration error.",
            "details": "Could not find the Python interpreter or the processing script. Please check server logs."
        }), 500

    except subprocess.CalledProcessError as e:
        # 捕获脚本内部执行出错的情况 (例如，无法连接RTMP流)
        # e.stderr 会包含脚本打印到标准错误的日志
        print(f"Subprocess failed with exit code {e.returncode}")
        print(f"Stderr: {e.stderr}")
        return jsonify({
            "error": "Failed to process the video stream.",
            "details": e.stderr.strip()
        }), 500

    except subprocess.TimeoutExpired:
        # 捕获脚本执行超时的情况
        print("Subprocess timed out.")
        return jsonify({"error": "Processing timed out. The RTMP stream may be slow or unavailable."}), 504

    finally:
        # 5. 无论成功与否，都尝试删除临时生成的图片文件，保持服务器整洁
        if os.path.exists(output_image_path):
            try:
                os.remove(output_image_path)
            except OSError as e:
                print(f"Error removing temporary file {output_image_path}: {e}")




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # 允许外部访问