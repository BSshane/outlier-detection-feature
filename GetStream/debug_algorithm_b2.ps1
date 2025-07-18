# debug_algorithm_b2.ps1 (已确认为 UTF-8 with BOM 格式)

# --- 新增：解决 PowerShell 终端输出的中文乱码问题 ---
[console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "--- 开始调试算法脚本 B ---" -ForegroundColor Yellow

# --- 配置区 (脚本路径需要保持) ---
$PATH_TO_PROCESSOR_SCRIPT = "D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\Dlib_face_recognition_from_camera\algorithm_b.py"
#$PATH_TO_PROCESSOR_SCRIPT = "D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\Dlib_face_recognition_from_camera\testFuckingHello.py"
# --- 模拟参数 ---
$TestVideoPath = "D:\0_MyBlog\test.mp4" # 使用你的本地视频文件测试
# $TestVideoPath = "0" # 或者使用摄像头测试
$TestOutputPath = ".\debug_output"

# 检查脚本是否存在
if (-not (Test-Path $PATH_TO_PROCESSOR_SCRIPT)) {
    Write-Error "找不到算法脚本: $PATH_TO_PROCESSOR_SCRIPT"
    exit 1
}
# 强制创建一个干净的输出文件夹
if (Test-Path $TestOutputPath) {
    Remove-Item -Path $TestOutputPath -Recurse -Force
}
New-Item -Path $TestOutputPath -ItemType Directory | Out-Null
Write-Host "已创建空的输出文件夹: $TestOutputPath" -ForegroundColor Green

Write-Host "将要执行的命令:"
# --- 【核心修改在这里】 ---
# 不再直接调用 python.exe，而是使用 conda run
$CommandToRun = "conda run -n GiteeADDlib python -u `"$PATH_TO_PROCESSOR_SCRIPT`" --stream-url `"$TestVideoPath`" --process-interval 5 --output-dir `"$TestOutputPath`""
# 在 debug_algorithm_b2.ps1 中修改这一行
#$CommandToRun = "conda run -n GiteeADDlib python -u `"$PATH_TO_PROCESSOR_SCRIPT`" --stream-url `"$TestVideoPath`" --process-interval 5 > $null"

# 如果你的算法有 --output-path, 就像这样:
# $CommandToRun = "conda run -n GiteeADDlib python -u `"$PATH_TO_PROCESSOR_SCRIPT`" --stream-url `"$TestVideoPath`" --output-path `"$TestOutputPath`""

Write-Host $CommandToRun -ForegroundColor Cyan

# --- 执行命令 ---
try {
    # 直接执行构建好的命令字符串
    Invoke-Expression $CommandToRun

    # # 检查输出文件是否已创建 (如果你的算法在调试时会创建文件)
    # if ($TestOutputPath -and (Test-Path $TestOutputPath)) {
    #     Write-Host "--- 调试成功 ---" -ForegroundColor Green
    #     Write-Host "脚本执行完毕，并成功生成了图片: $TestOutputPath"
    # } else {
    #     Write-Host "--- 脚本执行完毕 ---" -ForegroundColor Green
    #     Write-Host "请检查终端输出是否有错误信息。"
    # }

    # 由于你的算法b是持续输出流，所以它正常情况下不会自己结束，上面的检查逻辑可能不会执行。
    # 只要不报错，就说明在正常运行。
    Write-Host "--- 算法脚本已启动 ---" -ForegroundColor Green
    Write-Host "请在终端观察算法的日志输出。按 Ctrl+C 停止。"


} catch {
    # 如果 Invoke-Expression 启动失败，或者命令本身有语法错误，会在这里捕获
    Write-Error "--- 调试失败 ---"
    Write-Error "算法脚本执行时发生错误:"
    Write-Error $_.Exception.Message
}
