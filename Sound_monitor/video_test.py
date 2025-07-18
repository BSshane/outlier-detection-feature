import argparse, subprocess, sys, queue, threading, datetime, numpy as np
import tensorflow as tf, tensorflow_hub as hub

# ————————————————— 模型初始化 ————————————————— #
print('[INFO] Loading YAMNet model ⏳')
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet.class_map_path().numpy()
class_names = tf.io.gfile.GFile(class_map_path).read().splitlines()
print('[INFO] YAMNet ready, total classes =', len(class_names))

# ——————————————— 危险关键词定义 ——————————————— #
DANGER_KEYWORDS = {
    'gunshot', 'gunfire',              # 枪声
    'screaming', 'scream',             # 尖叫
    'glass', 'shatter',                # 玻璃碎裂
    'children shouting',               # 孩子喊叫
    'explosion', 'bang',               # 爆炸
    'siren', 'alarm',                  # 警报
}

# ——————————————— 危险检测函数 ——————————————— #
def detect_danger(waveform, threshold=0.35):
    """返回 (是否危险, top_label, score)"""
    scores, embeddings, spectrogram = yamnet(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_index = int(mean_scores.argmax())
    top_label = class_names[top_index]
    top_score = mean_scores[top_index]
    is_danger = any(k in top_label.lower() for k in DANGER_KEYWORDS) and top_score > threshold
    return bool(is_danger), top_label, float(top_score)

# ——————————————— 拉流线程函数 ——————————————— #
def audio_capture(video_path, pcm_queue):
    """
    使用 ffmpeg 从视频文件中提取音频，转成16kHz单声道PCM数据放入队列
    """
    cmd = [
        'ffmpeg', '-loglevel', 'error',
        '-i', video_path,
        '-vn',               # 不处理视频，只提取音频
        '-ac', '1',          # 单声道
        '-ar', '16000',      # 采样率16kHz
        '-f', 's16le',       # 输出格式16-bit PCM little endian
        'pipe:1'
    ]
    print(f'[INFO] 启动 FFmpeg 读取视频音频流：{video_path}')
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=4096)
    except FileNotFoundError:
        print('[ERROR] FFmpeg 未安装或不可用，请检查路径。')
        sys.exit(1)

    try:
        while True:
            data = proc.stdout.read(32000)  # 1秒音频数据 16000 samples * 2 bytes
            if not data:
                err_msg = proc.stderr.read()
                print(f'\n[WARNING] 3秒内未收到音频数据，可能读取结束或音轨异常。\nFFmpeg 错误信息：\n{err_msg.decode(errors="ignore")}')
                break
            pcm_queue.put(data)
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()

# ——————————————— 主函数 ——————————————— #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='视频文件路径或 URL，如 scream.mp4')
    ap.add_argument('--threshold', type=float, default=0.35, help='危险判定阈值（默认 0.35）')
    args = ap.parse_args()

    print(f'[INFO] 正在读取视频音频流：{args.video}')
    pcm_q = queue.Queue(maxsize=30)
    threading.Thread(target=audio_capture, args=(args.video, pcm_q), daemon=True).start()

    print('[INFO] 开始音频分析…（按 Ctrl+C 停止）')
    while True:
        try:
            raw = pcm_q.get(timeout=3.0)
        except queue.Empty:
            print('[WARNING] 3秒内未收到音频数据，可能读取结束或音轨异常。')
            continue
        wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        is_danger, label, score = detect_danger(wav, threshold=args.threshold)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if is_danger:
            print(f'\033[91m[ALERT] {now} - {label} (score={score:.2f})\033[0m')
        else:
            print(f'[OK]    {now} - {label} (score={score:.2f})')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INFO] 已手动终止监听。')
        sys.exit(0)