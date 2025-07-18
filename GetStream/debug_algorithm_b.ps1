# debug_algorithm_b.ps1

Write-Host "--- 开始调试算法脚本 B (SP_VideoProcess.py) ---" -ForegroundColor Yellow

# --- 配置区 (请确保路径正确) ---

# 算法B的Python解释器路径 (来自你的代码)
$PATH_TO_FACE_RECOG_INTERPRETER = "C:\Users\renyi\miniconda3\envs\GiteeADDlib\python.exe"

# 算法B的脚本路径 (来自你的代码)
$PATH_TO_PROCESSOR_SCRIPT = "D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\Dlib_face_recognition_from_camera\algorithm_b.py"

# --- 模拟参数 ---

# 模拟一个输入的RTMP流地址
#$TestRtmpUrl = "rtmp://your-test-rtmp-server/live/streamkey" # <== 在这里填入一个可用的测试流
# 如果没有RTMP流，可以用本地摄像头 "0" 或视频文件路径来测试
# $TestRtmpUrl = "0"
$TestRtmpUrl ="D:\0_MyBlog\test.mp4"

# 模拟一个输出图片的路径
$TestOutputPath = ".\debug_capture.jpg" # 将截图保存在当前目录，方便查看

# 检查解释器和脚本是否存在
if (-not (Test-Path $PATH_TO_FACE_RECOG_INTERPRETER)) {
    Write-Error "CANNOT FIND INTERPRETER: $PATH_TO_FACE_RECOG_INTERPRETER"
    exit 1
}
if (-not (Test-Path $PATH_TO_PROCESSOR_SCRIPT)) {
    Write-Error "CANNOT FIND SCRIPT B: $PATH_TO_PROCESSOR_SCRIPT"
    exit 1
}

Write-Host "COMMANDS TO BE EXECUTED:"
Write-Host "$PATH_TO_FACE_RECOG_INTERPRETER `"$PATH_TO_PROCESSOR_SCRIPT`" --rtmp-url `"$TestRtmpUrl`" --output-path `"$TestOutputPath`"" -ForegroundColor Cyan

# --- 执行命令 ---
try {
    # 使用 & 调用操作符执行命令
    & $PATH_TO_FACE_RECOG_INTERPRETER $PATH_TO_PROCESSOR_SCRIPT --rtmp-url $TestRtmpUrl --output-path $TestOutputPath

    # 检查输出文件是否已创建
    if (Test-Path $TestOutputPath) {
        Write-Host "--- SUCCESS ---" -ForegroundColor Green
        Write-Host "PICTURES SAVED IN: $TestOutputPath"
    } else {
        Write-Host "--- WARNING ---" -ForegroundColor Yellow
        Write-Host "SCRIPT FINISHED,BUT PICTURES NOT FOUND: $TestOutputPath"
    }

} catch {
    # 如果脚本执行出错（例如，抛出异常），在这里捕获并打印错误信息
    Write-Error "---FAILED---"
    Write-Error "ERROR OCCURRED:"
    Write-Error $_.Exception.Message
}
