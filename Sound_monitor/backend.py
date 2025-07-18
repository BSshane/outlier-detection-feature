from flask import Flask, request, jsonify, render_template
import subprocess
import threading
import queue
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
# 新增flask-restx导入
from flask_restx import Api, Resource, fields
from flask_cors import CORS  # 添加CORS导入

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})
# 初始化flask-restx Api
api = Api(app, version='1.0', title='声音监控API', description='实时声音异常检测系统API文档', doc='/swagger/')

# 创建命名空间
ns = api.namespace('monitor', description='声音监控操作')

# 定义请求模型
start_model = api.model('StartRequest', {
    'rtmp_url': fields.String(required=True, description='RTMP流地址或流ID')
})

# 全局变量
audio_queue = queue.Queue(maxsize=10)
stream_active = False
current_stream = None

# 加载YAMNet模型
print("Loading YAMNet model...")
yamnet = hub.load('./Sound_monitor/model')
class_map_path = yamnet.class_map_path().numpy()
class_names = tf.io.gfile.GFile(class_map_path).read().splitlines()
print("Model loaded successfully")

# 危险关键词
DANGER_KEYWORDS = {
    'gunshot', 'gunfire', 'screaming', 'scream',
    'glass', 'shatter', 'explosion', 'bang', 'siren', 'alarm',
    'cough', 'coughing', 'snort' ,'children shouting','children playing' # 新增咳嗽相关关键词
}


def audio_stream_reader(rtmp_url):
    global stream_active

    command = [
        'ffmpeg',
        '-i', rtmp_url,
        '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'
    ]

    try:
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 6)
        print(f"Started reading audio from {rtmp_url}")

        while stream_active:
            raw_audio = pipe.stdout.read(16000 * 2)
            if not raw_audio:
                break
            if not audio_queue.full():
                audio_queue.put(raw_audio)
    except Exception as e:
        print(f"Error in audio stream reader: {e}")
    finally:
        pipe.terminate()
        print("Audio stream reader stopped")


def analyze_audio():
    start_time = time.time()
    while stream_active and (time.time() - start_time) < 5:  # 5秒超时
        if not audio_queue.empty():
            raw_audio = audio_queue.get()
            try:
                waveform = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                scores, _, _ = yamnet(waveform)
                mean_scores = tf.reduce_mean(scores, axis=0).numpy()
                top_index = int(mean_scores.argmax())
                top_label = class_names[top_index]
                top_score = mean_scores[top_index]
                is_danger = any(k in top_label.lower() for k in DANGER_KEYWORDS)

                return {
                    'label': top_label,
                    'score': float(top_score),
                    'is_danger': is_danger,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                print(f"Error analyzing audio: {e}")
                return {
                    'label': 'error',
                    'score': 0,
                    'is_danger': False,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'error': str(e)
                }
        time.sleep(0.1)
    return {'error': 'Analysis timeout (5s)'}


@app.route('/')
def index():
    return render_template('index.html')


# 重构路由为flask-restx资源
@ns.route('/start')
@api.doc(description='开始RTMP流音频分析，如果提供数字ID将自动转换为完整RTMP地址')
class StartAnalysis(Resource):
    @api.expect(start_model)
    @api.response(200, '分析成功启动')
    @api.response(400, '请求参数错误或分析已在运行')
    def post(self):
        global stream_active, current_stream

        data = api.payload
        if not data or 'rtmp_url' not in data:
            return {'error': 'RTMP URL is required'}, 400

        # 修改这里：如果输入是纯数字，自动转换为完整RTMP地址
        rtmp_url = data['rtmp_url']
        rtmp_url = f'rtmp://1.92.135.70:9090/live/{rtmp_url}'

        if stream_active:
            return {'error': 'Analysis is already active'}, 400

        stream_active = True
        current_stream = rtmp_url
        threading.Thread(target=audio_stream_reader, args=(rtmp_url,), daemon=True).start()

        return {
            'status': 'success',
            'message': f'Started analyzing {rtmp_url}',
            'stream_url': rtmp_url
        }

@ns.route('/analyze')
@api.doc(description='获取当前音频分析结果')
class GetAnalysis(Resource):
    @api.response(200, '返回分析结果')
    @api.response(400, '没有活动的分析流')
    def get(self):
        if not stream_active:
            return {'error': 'No active stream'}, 400

        result = analyze_audio()
        return jsonify(result)

@ns.route('/stop')
@api.doc(description='停止当前音频分析')
class StopAnalysis(Resource):
    @api.response(200, '分析成功停止')
    @api.response(400, '没有活动的分析')
    def post(self):
        global stream_active

        if not stream_active:
            return {'error': 'No active analysis'}, 400

        stream_active = False
        while not audio_queue.empty():
            audio_queue.get()

        return jsonify({'status': 'success', 'message': 'Analysis stopped'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9084, threaded=True)