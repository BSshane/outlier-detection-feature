# detect.py 函数修改

# 导入必要的库
import time
import requests
import cv2  # 确保cv2已导入
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import random
# 假设您的 runOpenpose 在 action_detect 文件夹下
import runOpenpose
import argparse

# 定义 app.py 接收帧的地址
APP_UPLOAD_URL = "http://127.0.0.1:5000/upload_frame"


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # --- Directories ---
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # --- Initialize ---
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # --- 模型加载 ---
    # [优化建议1: 使用更小的YOLO模型]
    # 如果您现在用的是 yolov5l.pt，尝试换成 yolov5s.pt 或 yolov5m.pt
    # 这会极大提升YOLO部分的检测速度。
    print("[INFO] 加载YOLOv5模型中...")
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()
    print(f'[INFO] 使用设备: {device} | CUDA 可用: {torch.cuda.is_available()}')

    print("[INFO] 加载摔倒检测模型中...")
    net = torch.jit.load(r'./action_detect/checkPoint/openpose.jit', map_location=device)
    action_net = torch.jit.load(r'./action_detect/checkPoint/action.jit', map_location=device)
    print("[INFO] 摔倒检测模型加载完成。")

    # --- Data loader ---
    # [优化建议2: 降低输入分辨率]
    # 将 --img-size 参数设置得更小，例如 640 或 320，可以显著提速。
    # imgsz = 320 # 直接在这里覆盖，或者通过命令行参数传入
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warmup
    if device.type != 'cpu':
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(img.half() if half else img)

    t0_global = time.time()
    frame_count = 0

    # [优化建议3: 跳帧检测]
    # 我们将YOLO检测和姿态检测的频率分开控制
    YOLO_INFERENCE_INTERVAL = 1  # 每 1 帧进行一次YOLO检测（可以设为2或3来提速）
    POSE_INFERENCE_INTERVAL = 10  # 每 10 帧进行一次姿态检测 (原为5，增加间隔可以减少卡顿)

    for path, img, im0s, vid_cap in dataset:
        # --- 性能分析计时器 ---
        t_start_frame = time.time()

        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t_after_preprocess = time.time()

        # --- YOLOv5 推理 ---
        # 只在指定间隔的帧上运行YOLO
        if frame_count % YOLO_INFERENCE_INTERVAL == 0:
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            # 将最新的检测结果缓存起来
            latest_pred = pred
        else:
            # 对于跳过的帧，使用上一次的检测结果
            pred = latest_pred if 'latest_pred' in locals() else []

        t_after_yolo = time.time()

        # --- 结果处理和绘制 ---
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            boxList = []
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # 始终绘制最新的检测框
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    # 只处理人类的框用于姿态检测
                    if names[int(cls)] == 'person':
                        boxList.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

        t_after_plot = time.time()

        # --- 姿态检测 (最耗时的部分) ---
        # 只在指定间隔的帧上，并且检测到了人，才运行姿态检测
        if frame_count % POSE_INFERENCE_INTERVAL == 0 and len(boxList) > 0:
            print(f'--- Frame {frame_count}: Running Pose Detection ---')
            # 这里我们假设 run_demo 会直接在 im0 上绘制结果
            # 如果它返回一个新的图像，你应该这样写: im0 = runOpenpose.run_demo(...)
            runOpenpose.run_demo(net, action_net, [im0], 256, device, boxList)

        t_after_pose = time.time()

        # --- 编码与发送 ---
        # [优化建议4: 降低JPEG质量]
        ret, jpeg_buffer = cv2.imencode('.jpg', im0, [cv2.IMWRITE_JPEG_QUALITY, 75])

        t_after_encode = time.time()

        if ret:
            try:
                # 使用 timeout 避免网络问题阻塞整个流程
                requests.post(APP_UPLOAD_URL, data=jpeg_buffer.tobytes(), headers={'Content-Type': 'image/jpeg'},
                              timeout=0.5)
            except requests.exceptions.RequestException as e:
                # 只在第一次连接失败时打印，避免刷屏
                if frame_count < POSE_INFERENCE_INTERVAL * 2:
                    print(f"Error connecting to backend: {e}")

        t_after_send = time.time()

        # --- 打印性能分析结果 ---
        print(f"--- Frame {frame_count} Performance (ms) ---")
        print(f"Pre-process:   {(t_after_preprocess - t_start_frame) * 1000:.1f}")
        print(f"YOLO Inference:{(t_after_yolo - t_after_preprocess) * 1000:.1f}")
        print(f"Plotting:      {(t_after_plot - t_after_yolo) * 1000:.1f}")
        print(f"Pose Detection:{(t_after_pose - t_after_plot) * 1000:.1f}  <--- 卡顿元凶在这里")
        print(f"JPEG Encoding: {(t_after_encode - t_after_pose) * 1000:.1f}")
        print(f"Network Send:  {(t_after_send - t_after_encode) * 1000:.1f}")
        print(f"------------------------------------")
        total_time = t_after_send - t_start_frame
        print(f"Total Time:    {total_time * 1000:.1f} ms | FPS: {1 / total_time:.1f}")
        print(f"====================================\n")

    print(f'Done. ({time.time() - t0_global:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='rtmp://1.92.135.70:9090/live/2',help='source') # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='rtmp://1.92.135.70:9090/live/1',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', default=0, type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()