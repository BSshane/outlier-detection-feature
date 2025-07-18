import argparse
import time
from pathlib import Path
from torch import from_numpy, jit

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import datetime
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, \
    set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import runOpenpose
import requests # 新增：用于发送HTTP请求

# 定义 app.py 接收帧的地址
APP_UPLOAD_URL = "http://127.0.0.1:5000/upload_frame" # 确保这个地址和app.py运行的地址一致

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    # Directories (保留原有保存标签的逻辑，但不再保存图片帧)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    # output_dir 和其创建现在不再需要，因为图片不再保存到本地
    # output_dir = save_dir / 'output'
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device) #加载模型
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()
    print(f'[INFO] 使用设备: {device} | CUDA 可用: {torch.cuda.is_available()}')

    # Load fall detection models
    print("[INFO] 加载摔倒检测模型中...")
    net = torch.jit.load(r'./action_detect/checkPoint/openpose.jit', map_location=device) #提取人体关键点
    action_net = torch.jit.load(r'./action_detect/checkPoint/action.jit', map_location=device) #关键点分类动作
    print("[INFO] 摔倒检测模型加载完成。")

    # Data loader
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True # 如果您想在 detect.py 端也显示画面，可以保留此行
        dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warmup
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    t0 = time.time()
    frame_count=0

    YOLO_INFERENCE_INTERVAL = 1  # 每 1 帧进行一次YOLO检测（可以设为2或3来提速）
    POSE_INFERENCE_INTERVAL = 10 # 每 10 帧进行一次姿态检测 (原为5，增加间隔可以减少卡顿)

    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if frame_count % YOLO_INFERENCE_INTERVAL == 0:
            # 每帧都执行。将图像送入 YOLOv5 模型，进行目标检测。
            pred = model(img, augment=opt.augment)[0]
            # 对检测结果进行后处理，去掉重叠的、置信度低的框。
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            latest_pred = pred
            # 遍历检测到的结果
        else:
            # 对于跳过的帧，使用上一次的检测结果
            pred = latest_pred if 'latest_pred' in locals() else []

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            boxList = [] #在每一帧开始时，清空 boxList。这个列表用于存放当前帧检测到的人的边界框
            p = Path(p)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):#只有在画面中检测到物体时，才执行以下操作。
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f'{n} {names[int(c)]}s, '

                if save_txt:
                    for *xyxy, conf, cls in reversed(det):# 遍历当前帧检测到的每一个物体。
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                for *xyxy, conf, cls in reversed(det):# 遍历当前帧检测到的每一个物体。
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # 这是一个逻辑错误点。boxList 在这个循环内部才被填充，但这个判断条件却在填充之前。
                    # 这会导致 plot_one_box 和 boxList.append 永远不会被执行。
                    # 我推断您的意图是：在每一帧都绘制边界框并填充 boxList，然后每5帧才进行姿态检测。
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    boxList.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

            # 每 5 帧进行一次人体姿态检测
            if frame_count % 5 == 0 and len(boxList) > 0:
                for box in boxList:
                    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    x = c2[0] - c1[0]
                    y = c2[1] - c1[1]
                    if x / y >= 0.8:  # 比例 > 0.8 可能摔倒
                        print(f'第 {frame_count} 帧：进行人体姿态检测')
                        runOpenpose.run_demo(net, action_net, [im0], 256, device, boxList)
                        break


            # 核心修改：将处理后的帧编码为JPEG并发送给 app.py
            ret, jpeg = cv2.imencode('.jpg', im0)
            if ret:
                try:
                    requests.post(APP_UPLOAD_URL, data=jpeg.tobytes(), headers={'Content-Type': 'image/jpeg'})
                except requests.exceptions.ConnectionError as e:
                    print(f"Error: Could not connect to app.py at {APP_UPLOAD_URL}. Is app.py running? {e}")
                except Exception as e:
                    print(f"Error sending frame to app.py: {e}")
            else:
                print("Error: Could not encode image to JPEG.")

    # cv2.destroyAllWindows()
    print(f'Done. ({time.time() - t0:.3f}s)')


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