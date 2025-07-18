import cv2
import time

rtmp_url = 'rtmp://1.92.135.70:9090/live/1'

print(f"尝试打开 RTMP 流: {rtmp_url}")
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print(f"错误: 无法打开 RTMP 流 '{rtmp_url}'。请检查流地址、网络连接和防火墙。")
else:
    print("成功打开 RTMP 流。正在尝试读取帧...")
    # 尝试读取几帧
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"错误: 无法读取帧 {i+1}。")
            break
        print(f"成功读取帧 {i+1}，帧尺寸: {frame.shape}")
        time.sleep(0.1) # 等待一小段时间

    print("测试完成。")
    cap.release() # 释放视频捕获对象