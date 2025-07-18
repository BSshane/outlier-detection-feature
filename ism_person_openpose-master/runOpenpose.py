import argparse
import cv2
import numpy as np
import torch
from openpose_modules.keypoints import extract_keypoints, group_keypoints
from openpose_modules.pose import Pose
from action_detect.detect import action_detect
import os
from math import ceil, floor
from utils.contrastImg import coincide
#
#
# # ######################### 核心修改点 #########################
# # 移除了此处的全局 device 定义，让 device 通过函数参数传递
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #############################################################
#
# class ImageReader(object):
#     def __init__(self, file_names):
#         self.file_names = file_names
#         self.max_idx = len(file_names)
#
#     def __iter__(self):
#         self.idx = 0
#         return self
#
#     def __next__(self):
#         if self.idx == self.max_idx:
#             raise StopIteration
#         img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
#         if img.size == 0:
#             raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
#         self.idx += 1
#         return img
#
#
# class VideoReader(object):
#     def __init__(self, file_name, code_name):
#         self.file_name = file_name
#         self.code_name = str(code_name)
#         try:
#             self.file_name = int(file_name)
#         except ValueError:
#             pass
#
#     def __iter__(self):
#         self.cap = cv2.VideoCapture(self.file_name)
#         if not self.cap.isOpened():
#             raise IOError('Video {} cannot be opened'.format(self.file_name))
#         return self
#
#     def __next__(self):
#         was_read, img = self.cap.read()
#         if not was_read:
#             raise StopIteration
#         cv2.putText(img, self.code_name, (5, 35),
#                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
#         return img
#
#
# # ######################### 核心修改点 #########################
# # 修复了 TypeError: unsupported operand type(s) for -: 'Tensor' and 'tuple'
# def normalize(img, img_mean, img_scale, device):
#     img = torch.tensor(img, dtype=torch.float32, device=device)
#     mean = torch.tensor(img_mean, dtype=torch.float32, device=device)
#     scale = torch.tensor(img_scale, dtype=torch.float32, device=device)
#     img = (img - mean) / scale
#     return img
#
#
# # #############################################################
#
# def pad_width(img, stride, pad_value, min_dims):
#     h, w, _ = img.shape
#     h = min(min_dims[0], h)
#     min_dims[0] = ceil(min_dims[0] / float(stride)) * stride
#     min_dims[1] = max(min_dims[1], w)
#     min_dims[1] = ceil(min_dims[1] / float(stride)) * stride
#     pad = [
#         int(floor((min_dims[0] - h) / 2.0)),
#         int(floor((min_dims[1] - w) / 2.0)),
#         int(min_dims[0] - h - floor((min_dims[0] - h) / 2.0)),
#         int(min_dims[1] - w - floor((min_dims[1] - w) / 2.0))
#     ]
#     # 使用 PyTorch 进行填充以支持 GPU 上的操作
#     # 注意：需要将 img 转换为 CHW 格式进行填充，然后再转回 HWC
#     img_tensor = torch.from_numpy(img).permute(2, 0, 1)
#     padded_img_tensor = torch.nn.functional.pad(
#         img_tensor,
#         (pad[1], pad[3], pad[0], pad[2]),  # left, right, top, bottom
#         mode='constant',
#         value=pad_value
#     )
#     padded_img = padded_img_tensor.permute(1, 2, 0).numpy()
#     return padded_img, pad
#
#
# def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, device,
#                pad_value=0, img_mean=(128, 128, 128), img_scale=1 / 256):
#     height, width, _ = img.shape
#     scale = net_input_height_size / height
#
#     scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#
#     # 注意：此处 normalize 会将 numpy 数组转换为 device 上的 tensor
#     scaled_img = normalize(scaled_img, img_mean, img_scale, device)
#
#     min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
#
#     # 填充操作现在也需要处理 Tensor
#     h, w, _ = scaled_img.shape
#     h = min(min_dims[0], h)
#     min_dims[0] = ceil(min_dims[0] / float(stride)) * stride
#     min_dims[1] = max(min_dims[1], w)
#     min_dims[1] = ceil(min_dims[1] / float(stride)) * stride
#     pad = [
#         int(floor((min_dims[0] - h) / 2.0)),
#         int(floor((min_dims[1] - w) / 2.0)),
#         int(min_dims[0] - h - floor((min_dims[0] - h) / 2.0)),
#         int(min_dims[1] - w - floor((min_dims[1] - w) / 2.0))
#     ]
#     padded_img = torch.nn.functional.pad(
#         scaled_img.permute(2, 0, 1),
#         (pad[1], pad[3], pad[0], pad[2]),
#         mode='constant',
#         value=pad_value
#     ).permute(1, 2, 0)
#
#     tensor_img = padded_img.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW, add batch dim
#
#     with torch.no_grad():
#         stages_output = net(tensor_img)
#
#     stage2_heatmaps = stages_output[-2]
#     heatmaps = torch.nn.functional.interpolate(
#         stage2_heatmaps, scale_factor=upsample_ratio, mode='bicubic', align_corners=False)
#     heatmaps = heatmaps.squeeze(0).permute(1, 2, 0).cpu().numpy()
#
#     stage2_pafs = stages_output[-1]
#     pafs = torch.nn.functional.interpolate(
#         stage2_pafs, scale_factor=upsample_ratio, mode='bicubic', align_corners=False)
#     pafs = pafs.squeeze(0).permute(1, 2, 0).cpu().numpy()
#
#     return heatmaps, pafs, scale, pad
#
#
# # ######################### 核心修改点 #########################
# # 将 'cpu' bool 参数替换为 'device' 对象
# def run_demo(net, action_net, image_provider, height_size, device, boxList):
#     # 模型设置
#     net = net.to(device).eval()
#     action_net = action_net.to(device).eval()
#
#     stride = 8
#     upsample_ratio = 4
#     num_keypoints = Pose.num_kpts
#
#     orig_img = image_provider[0]
#     final_img_to_draw = orig_img.copy()  # 创建一个副本用于最终绘制
#
#     for box in boxList:
#         # 1. 裁剪出 ROI (Region of Interest)
#         x1, y1, x2, y2 = box
#         # 添加一些边距(padding)以确保完整的人体被包含
#         pad_x = int((x2 - x1) * 0.1)
#         pad_y = int((y2 - y1) * 0.1)
#         crop_x1 = max(0, x1 - pad_x)
#         crop_y1 = max(0, y1 - pad_y)
#         crop_x2 = min(orig_img.shape[1], x2 + pad_x)
#         crop_y2 = min(orig_img.shape[0], y2 + pad_y)
#
#         # 如果裁剪区域无效，则跳过
#         if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
#             continue
#
#         cropped_img = orig_img[crop_y1:crop_y2, crop_x1:crop_x2]
#         # 2. 对裁剪出的小图进行姿态估计
#         heatmaps, pafs, scale, pad = infer_fast(net, cropped_img, height_size, stride, upsample_ratio, device)
#
#         # 3. 后处理（这部分逻辑与您原有的类似，但现在是在小图上操作）
#         all_keypoints_by_type = []
#         total_keypoints_num = 0
#         for kpt_idx in range(num_keypoints):
#             total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
#                                                      total_keypoints_num)
#
#         pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
#         if len(pose_entries) == 0:
#             continue
#
#         # 4. 将关键点坐标转换回原始大图的坐标系
#         for kpt_id in range(all_keypoints.shape[0]):
#             # 先从小图的缩放坐标转为小图的原始坐标
#             all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
#             all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
#             # 再加上裁剪框的左上角偏移，转为大图坐标
#             all_keypoints[kpt_id, 0] += crop_x1
#             all_keypoints[kpt_id, 1] += crop_y1
#
#         # 通常一个小图里只有一个人，我们处理找到的第一个姿态即可
#         entry = pose_entries[0]
#         pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
#         for kpt_id in range(num_keypoints):
#             if entry[kpt_id] != -1.0:
#                 pose_keypoints[kpt_id, 0] = int(all_keypoints[int(entry[kpt_id]), 0])
#                 pose_keypoints[kpt_id, 1] = int(all_keypoints[int(entry[kpt_id]), 1])
#
#         pose = Pose(pose_keypoints, entry[18])
#
#         # 5. 动作识别与绘制
#         if len(pose.getKeyPoints()) >= 5: # 放宽关键点数量要求
#             pose.img_pose = pose.draw(final_img_to_draw, is_save=False, show_draw=False) # 在副本上绘制
#             crown_proportion = pose.bbox[2] / pose.bbox[3] if pose.bbox[3] > 0 else 0
#             pose = action_detect(action_net, pose, crown_proportion, device)
#
#             color = (0, 0, 255) if pose.pose_action == 'fall' else (0, 255, 0)
#             cv2.rectangle(final_img_to_draw, (pose.bbox[0], pose.bbox[1]),
#                           (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), color, thickness=2)
#             cv2.putText(final_img_to_draw, f'state: {pose.pose_action}', (pose.bbox[0], pose.bbox[1] - 10),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
#
#     # 6. 将绘制了所有姿态的图像与原始图像融合（可选，但效果好）
#     final_img_to_show = cv2.addWeighted(orig_img, 0.4, final_img_to_draw, 0.6, 0)
#
#     # 注意：这里的 image_provider[0] = final_img_to_show 是为了让 detect.py 能拿到最终绘制的图
#     image_provider[0] = final_img_to_show
#
#     # i = 0
#     # for img in image_provider:
#     #     orig_img = img.copy()
#     #     if i % 1 == 0:
#     #         heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, device)
#     #
#     #         total_keypoints_num = 0
#     #         all_keypoints_by_type = []
#     #         for kpt_idx in range(num_keypoints):
#     #             total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
#     #                                                      total_keypoints_num)
#     #
#     #         pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
#     #         for kpt_id in range(all_keypoints.shape[0]):
#     #             all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
#     #             all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
#     #         current_poses = []
#     #         for n in range(len(pose_entries)):
#     #             if len(pose_entries[n]) == 0:
#     #                 continue
#     #             pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
#     #             for kpt_id in range(num_keypoints):
#     #                 if pose_entries[n][kpt_id] != -1.0:
#     #                     pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
#     #                     pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
#     #             pose = Pose(pose_keypoints, pose_entries[n][18])
#     #             posebox = (int(pose.bbox[0]), int(pose.bbox[1]), int(pose.bbox[0]) + int(pose.bbox[2]),
#     #                        int(pose.bbox[1]) + int(pose.bbox[3]))
#     #             if boxList:
#     #                 coincideValue = coincide(boxList, posebox)
#     #                 print(posebox)
#     #                 print('coincideValue:' + str(coincideValue))
#     #                 if len(pose.getKeyPoints()) >= 10 and coincideValue >= 0.3 and pose.lowerHalfFlag < 3:
#     #                     current_poses.append(pose)
#     #             else:
#     #                 current_poses.append(pose)
#     #         for pose in current_poses:
#     #             pose.img_pose = pose.draw(img, is_save=True, show_draw=True)
#     #             crown_proportion = pose.bbox[2] / pose.bbox[3] if pose.bbox[3] > 0 else 0
#     #             pose = action_detect(action_net, pose, crown_proportion, device)  # 确保 action_detect 也接收 device
#     #
#     #             if pose.pose_action == 'fall':
#     #                 cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
#     #                               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255), thickness=3)
#     #                 cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
#     #                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
#     #             else:
#     #                 cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
#     #                               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
#     #                 cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
#     #                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
#     #
#     #         img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
#     #         cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
#     #         cv2.waitKey(1)
#     #     i += 1
#     # 注意：如果这是一个持续的流，你可能不希望在这里销毁窗口
#     # cv2.destroyAllWindows()
#     cv2.destroyAllWindows()
#
# def detect_main(video_name=''):
#     parser = argparse.ArgumentParser(
#         description='''Lightweight human pose estimation python demo.
#                            This is just for quick results preview.
#                            Please, consider c++ demo for the best performance.''')
#     parser.add_argument('--checkpoint-path', type=str, default='openpose.jit',
#                         help='path to the checkpoint')
#     parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
#     parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
#     parser.add_argument('--images', nargs='+',
#                         default='D:\\project\\ism_person_openpose\\data\\pics',
#                         help='path to input image(s)')
#     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
#                         help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--code_name', type=str, default='None', help='the name of video')
#     args = parser.parse_args()
#
#     if video_name != '':
#         args.code_name = video_name
#
#     if args.video == '' and args.images == '':
#         raise ValueError('Either --video or --image has to be provided')
#
#     net = torch.jit.load(r'.\action_detect\checkPoint\openpose.jit').to(device)
#     action_net = torch.jit.load(r'.\action_detect\checkPoint\action.jit').to(device)
#
#     if args.video != '':
#         frame_provider = VideoReader(args.video, args.code_name)
#     else:
#         images_dir = []
#         if os.path.isdir(args.images):
#             for img_dir in os.listdir(args.images):
#                 images_dir.append(os.path.join(args.images, img_dir))
#             frame_provider = ImageReader(images_dir)
#         else:
#             img = cv2.imread(args.images, cv2.IMREAD_COLOR)
#             frame_provider = [img]
#
#     run_demo(net, action_net, frame_provider, args.height_size, args.device == 'cpu', [])
#
# if __name__ == '__main__':
#     detect_main()


# runOpenpose.py (高性能重构版)

import cv2
import numpy as np
import torch
from math import ceil, floor

# 导入您项目中的 Pose 类和后处理函数
# 请根据您的项目结构调整这里的导入路径
from openpose_modules.pose import Pose
from openpose_modules.keypoints import extract_keypoints, group_keypoints
from action_detect.detect import action_detect  # 假设 action_detect 在这里


# --- 辅助函数，进行了优化 ---

def normalize(img_tensor, img_mean, img_scale):
    """直接在Tensor上进行标准化，避免数据来回拷贝"""
    # img_mean 和 img_scale 已经是tensor，直接计算
    img_tensor = (img_tensor - img_mean) / img_scale
    return img_tensor


def pad_image_to_stride(img_tensor, stride, pad_value):
    """使用PyTorch的pad函数对Tensor进行填充，效率更高"""
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    h_new = ceil(h / stride) * stride
    w_new = ceil(w / stride) * stride
    pad_h = h_new - h
    pad_w = w_new - w
    # (pad_left, pad_right, pad_top, pad_bottom)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    padded_tensor = torch.nn.functional.pad(img_tensor, padding, mode='constant', value=pad_value)
    return padded_tensor, padding


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, device):
    """
    推理函数，现在接收numpy格式的图像，内部完成所有转换。
    这个函数现在只处理裁剪后的小图，速度会飞快。
    """
    height, width, _ = img.shape
    scale = net_input_height_size / height

    # 1. 缩放
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 2. 转换为Tensor并移动到GPU
    # BGR -> RGB, HWC -> CHW, to float, to device
    tensor_img = torch.from_numpy(scaled_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor_img = tensor_img.flip(1)  # BGR to RGB if needed, assumes cv2 BGR format

    # 3. 标准化
    img_mean = torch.tensor([128, 128, 128], device=device).view(1, 3, 1, 1)
    img_scale = torch.tensor([1 / 256.0], device=device).view(1, 1, 1, 1)
    tensor_img = normalize(tensor_img, img_mean, img_scale)

    # 4. 填充
    padded_tensor, pad = pad_image_to_stride(tensor_img, stride, 0)

    # 5. 推理
    with torch.no_grad():
        stages_output = net(padded_tensor)

    # 6. 后处理和上采样
    heatmaps = stages_output[-2]
    pafs = stages_output[-1]

    if upsample_ratio > 1:
        heatmaps = torch.nn.functional.interpolate(heatmaps, scale_factor=upsample_ratio, mode='bicubic',
                                                   align_corners=False)
        pafs = torch.nn.functional.interpolate(pafs, scale_factor=upsample_ratio, mode='bicubic', align_corners=False)

    # 7. 移回CPU并转为numpy
    heatmaps = heatmaps.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pafs = pafs.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 将pad的格式统一为 (top, bottom, left, right)
    pad_t, pad_b, pad_l, pad_r = pad[2], pad[3], pad[0], pad[1]

    return heatmaps, pafs, scale, (pad_t, pad_b, pad_l, pad_r)


# ######################### 核心修改点：重构run_demo #########################
def run_demo(net, action_net, image_provider, height_size, device, boxList):
    # 确保模型在正确的设备上并处于评估模式
    net = net.to(device).eval()
    action_net = action_net.to(device).eval()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    orig_img = image_provider[0]
    # 创建一个副本用于绘制，避免污染原始图像
    final_img_to_draw = orig_img.copy()

    # 遍历YOLO检测到的每一个人体框
    for box in boxList:
        # --- 1. 裁剪ROI (Region of Interest) ---
        x1, y1, x2, y2 = box

        # 如果框无效，则跳过
        if y2 <= y1 or x2 <= x1:
            continue

        cropped_img = orig_img[y1:y2, x1:x2]

        # 如果裁剪区域是空的，则跳过
        if cropped_img.size == 0:
            continue

        # --- 2. 对裁剪出的小图进行姿态估计 (这是性能提升的关键) ---
        heatmaps, pafs, scale, pad = infer_fast(net, cropped_img, height_size, stride, upsample_ratio, device)

        # --- 3. 提取和组合关键点 (在小图的热图上操作) ---
        all_keypoints_by_type = []
        total_keypoints_num = 0
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        if len(pose_entries) == 0:
            continue

        # --- 4. 将关键点坐标转换回原始大图的坐标系 ---
        # 遍历所有找到的关键点
        for kpt_id in range(all_keypoints.shape[0]):
            # a. 从热图坐标转换回填充后的小图坐标
            all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] * stride / upsample_ratio
            all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] * stride / upsample_ratio
            # b. 减去填充(padding)的偏移
            all_keypoints[kpt_id, 0] -= pad[2]  # pad_left
            all_keypoints[kpt_id, 1] -= pad[0]  # pad_top
            # c. 除以缩放比例，转回裁剪前的小图坐标
            all_keypoints[kpt_id, 0] /= scale
            all_keypoints[kpt_id, 1] /= scale
            # d. 加上裁剪框的左上角偏移(x1, y1)，转为原始大图坐标
            all_keypoints[kpt_id, 0] += x1
            all_keypoints[kpt_id, 1] += y1

        # --- 5. 组装姿态并进行动作识别 ---
        # 通常一个裁剪框里只有一个人，我们处理找到的第一个姿态即可
        entry = pose_entries[0]
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if entry[kpt_id] != -1.0:  # 如果这个关键点被找到了
                point_id = int(entry[kpt_id])
                pose_keypoints[kpt_id, 0] = int(all_keypoints[point_id, 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[point_id, 1])

        pose = Pose(pose_keypoints, entry[18])

        # 如果检测到的关键点足够多，就进行动作识别
        if pose.num_kpts >= 5:
            # 动作识别也应该在GPU上运行
            # 计算人体框的宽高比
            crown_proportion = pose.bbox[2] / pose.bbox[3] if pose.bbox[3] > 0 else 0
            # 将计算出的宽高比传递给 action_detect 函数
            pose = action_detect(action_net, pose, crown_proportion, device)

            # --- 6. 在副本图像上绘制结果 ---
            pose.draw(final_img_to_draw)  # 在副本上绘制骨骼
            color = (0, 0, 255) if pose.pose_action == 'fall' else (0, 255, 0)
            cv2.rectangle(final_img_to_draw, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), color, thickness=2)
            cv2.putText(final_img_to_draw, f'state: {pose.pose_action}', (pose.bbox[0], pose.bbox[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # --- 7. 将绘制结果更新回 image_provider ---
    # 让 detect.py 能拿到最终绘制了所有信息的图
    # 注意：这里直接修改了列表中的元素，可以生效
    image_provider[0] = final_img_to_draw


# 如果这个文件被独立运行，则保留其原有功能
if __name__ == '__main__':
    # ... 您原有的 if __name__ == '__main__': 部分的代码可以放在这里 ...
    # 例如 argparse 和调用 detect_main()
    pass
