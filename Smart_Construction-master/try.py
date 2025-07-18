import torch
import os

# 强制 PyTorch 使用特定 CUDA 版本（可选）
# os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"

# 检查 CUDA 可用性
print("CUDA 可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())

# 创建简单模型并测试
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(8*224*224, 2)
).to(device)

# 测试前向传播
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    y = model(x)
print("模型输出形状:", y.shape)