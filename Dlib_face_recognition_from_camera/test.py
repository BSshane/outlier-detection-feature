import torch

print("是否支持 CUDA：", torch.cuda.is_available())
print("GPU 数量：", torch.cuda.device_count())

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print("当前 GPU 编号：", current_device)
    print("GPU 名称：", torch.cuda.get_device_name(current_device))
    print("CUDA 运行时版本：", torch.version.cuda)
else:
    print("未检测到 GPU 或未安装 CUDA 版本 PyTorch。")