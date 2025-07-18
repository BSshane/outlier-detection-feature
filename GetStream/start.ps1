# PowerShell 脚本出错时立即退出
$ErrorActionPreference = "Stop"

# --- 重要: 请修改成你自己的绝对路径 ---
# 获取当前脚本所在的目录的绝对路径
$ProjectRoot = $PSScriptRoot
# 或者手动指定
# $ProjectRoot = "C:\Users\YourName\Documents\my_stream_project"

$VENV_A_PYTHON = "C:\Users\renyi\miniconda3\envs\GiteeAD\python.exe"
$VENV_B_PYTHON = "C:\Users\renyi\miniconda3\envs\GiteeADDlib\python.exe"

# --- 脚本路径 (通常不需要修改) ---
$SCRIPT_A_PATH = "D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\GetStream\frontend\flask_backend\app.py"
$SCRIPT_B_PATH = "D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\Dlib_face_recognition_from_camera\algorithm_b.py"

# --- 在这里定义你的视频流来源 ---
# $VideoStreamSource = "0"  # 使用本地摄像头
$VideoStreamSource = "D:\0_MyBlog\test.mp4" # 使用视频文件

# --- 在这里定义处理间隔 ---
$ProcessInterval = "5"

Write-Host "启动中... 视频源: $VideoStreamSource"
Write-Host "请在浏览器中访问 http://localhost:8080"
Write-Host "在 PowerShell 窗口中按 Ctrl+C 停止。"

# --- 【核心修改】 ---
# 使用具名参数 --stream-url 和 --process-interval 调用算法脚本
& $VENV_B_PYTHON -u $SCRIPT_B_PATH --stream-url $VideoStreamSource --process-interval $ProcessInterval | & $VENV_A_PYTHON -u $SCRIPT_A_PATH
