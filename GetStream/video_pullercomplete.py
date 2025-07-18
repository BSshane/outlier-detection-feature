from flask import Flask, Response, render_template
import cv2
import time

app = Flask(__name__)

def process_stream(stream_url, timeout=10):
    cap = cv2.VideoCapture(stream_url)
    start_time = time.time()
    while not cap.isOpened():
        if time.time() - start_time > timeout:
            raise RuntimeError(f"无法打开视频流: {stream_url} (超时{timeout}s)")
        time.sleep(0.1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 5
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                frame_count += 1
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<stream_id>')
def video_feed(stream_id):
    stream_url = f'rtmp://1.92.135.70:9090/live/{stream_id}'
    try:
        return Response(process_stream(stream_url),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except RuntimeError as e:
        app.logger.error(f"视频流处理失败: {e}")
        return f"视频流打开失败: {e}", 500
    except Exception as e:
        app.logger.error(f"未知错误: {e}")
        return "视频流处理失败", 500

if __name__ == '__main__':
    app.run(debug=True)