import numpy as np
import torch  # 导入 torch 以便使用 torch.no_grad()


def action_detect(net, pose, crown_proportion, device):
    """
    摔倒/正常 姿态识别
    :param net: action 识别模型 (已在 device 上)
    :param pose: 单人姿态
    :param crown_proportion: 单人姿态的宽高比
    :param device: 计算设备 (例如 'cuda:0' 或 'cpu')
    :return: 更新了动作状态的 pose 对象
    """
    # 确保模型处于评估模式
    net.eval()

    # 你的原始逻辑来准备数据
    # maxHeight = pose.keypoints.max()
    # minHeight = pose.keypoints.min()

    # 这里假设 pose.img_pose 是一个有效的 numpy 数组
    if pose.img_pose is None or pose.img_pose.size == 0:
        pose.pose_action = 'unknown'
        return pose

    img = pose.img_pose.reshape(-1)
    img = img / 255.0  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    # ######################### 核心修复点 #########################
    # 将 NumPy 数组转换为 Tensor，并将其发送到正确的 `device`
    # 之前是 .cpu()，现在我们使用传入的 `device` 变量
    img = torch.from_numpy(img[None, :]).to(device)
    # #############################################################

    # 使用 torch.no_grad() 来进行推理，这可以提高性能并减少内存使用
    with torch.no_grad():
        predect = net(img)

    # argmax 操作可以在 GPU 上完成，然后再将结果移动到 CPU
    action_id = int(torch.argmax(predect, dim=1).item())

    # 将 predect 移动到 CPU 以便进行后续的 numpy 操作
    predect = predect.cpu()

    # 你的原始计算逻辑
    possible_rate = 0.6 * predect[:, action_id] + 0.4 * (crown_proportion - 1)
    possible_rate = possible_rate.detach().numpy()[0]

    if possible_rate > 0.55:
        pose.pose_action = 'fall'
        if possible_rate > 1:
            possible_rate = 1
        pose.action_fall = possible_rate
        pose.action_normal = 1 - possible_rate
    else:
        pose.pose_action = 'normal'
        if possible_rate >= 0.5:
            pose.action_fall = 1 - possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate

    return pose

