import os
import sys
import subprocess
import time
from typing import Generator


# --- 配置区 ---
# 写死虚拟环境路径

# Windows
PYTHON_B_PATH = r"C:/Users/renyi/miniconda3/envs/GiteeADDlib/python.exe"
# PYTHON_C_PATH = os.path.join(PROJECT_ROOT, "venv_C", "Scripts", "python.exe")

SCRIPT_B_PATH = r"D:\1_MyPrograms\7SummerPrac\GiteeProject\AbnormalDetection\Video_Monitoring_Project\Dlib_face_recognition_from_camera\algorithm_b.py"
INPUT_VIDEO = r"D:\0_MyBlog\test.mp4"
# SCRIPT_C_PATH = os.path.join(PROJECT_ROOT, "algorithm_c", "main_c.py")





def get_jpeg_stream_from_algorithm(python_interpreter: str, script_path: str, video_source: str) -> Generator[
    bytes, None, None]:
    """
    通过子进程调用流式算法，并实时产出纯净的JPEG数据流。

    :param python_interpreter: 目标虚拟环境的Python解释器路径
    :param script_path: 要执行的流式算法脚本路径
    :param video_source: 视频源（文件路径或摄像头ID）
    :return: 一个生成器，每次产出(yield)一帧的纯JPEG二进制数据
    """
    # conda_exe_path=r"C:\Users\renyi\miniconda3\Scripts\conda.exe"
    conda_exe_path=r"C:\Users\renyi\miniconda3\condabin\conda.bat"
    env_name="GiteeADDlib"
    # command = [
    #     conda_exe_path,
    #     "run",
    #     "-n", env_name,
    #     PYTHON_B_PATH,  # conda run 会自动找到该环境下的 python
    #     script_path,
    #     '--stream-url', video_source
    # ]
    # print(f"[*] Starting streaming process with command: {' '.join(command)}")

    command_str = (
        f'"{conda_exe_path}" run -n {env_name} python "{SCRIPT_B_PATH}" '
        f'--stream-url "{INPUT_VIDEO}"'
    )

    print(f"[*] Starting streaming process with command: {command_str}")

    # 使用 subprocess.Popen 启动子进程，因为它不会阻塞
    # stdout=subprocess.PIPE: 将子进程的标准输出重定向到一个管道，我们可以从中读取
    # stderr=subprocess.PIPE: 同样捕获标准错误，用于调试
    # bufsize=1: 行缓冲模式，确保我们能尽快收到数据
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,bufsize=1)

    try:
        process = subprocess.Popen(
            command_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,  # 关键：让系统shell来解释整个命令
            creationflags=subprocess.CREATE_NO_WINDOW  # （可选）在Windows上隐藏弹出的黑窗
        )
    except FileNotFoundError:
        print(f"[ERROR] Cannot find command: {conda_exe_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to start subprocess: {e}")
        return None
    try:
        boundary = b'--fram\r'
        header_separater=b'\r\r'
        buffer=b''
        while True:
            if process.poll() is not None:# 检查子进程是否在运行
                # 子进程已退出
                print(f"[!] Subprocess terminated with code {process.poll()}.")
                # 读取并打印所有剩余的错误信息
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    print("-------stderror------")
                    print(stderr_output)
                break
            # 从管道读取数据模块
            chunk = process.stdout.read(1024)
            if not chunk:
                time.sleep(0.01) # 避免CPU空转
                continue

            buffer+=chunk

            # 在缓冲流查找并处理完整的帧
            while boundary in buffer:
                start=buffer.find(boundary)
                end=buffer.find(boundary, start+len(boundary))

                if end==-1:
                    break #此时说明帧不完整
                frame = buffer[start:end]
                buffer = buffer [end:] # 更新缓冲区
                header_end=frame.find(header_separater)
                # 在帧部分中，找到头部和图像数据的分隔符
                if header_end!=-1:
                    # 提取纯净的JPEG数据 在header_separater和\r之间
                    image_data=frame[header_end+len(header_separater):-2]
                    yield image_data
    finally:
        print("[*] Cleaning up subprocess...")
        if process.poll() is None:  # 如果进程仍在运行
            process.terminate()  # 发送终止信号
            try:
                process.wait(timeout=5)  # 等待5秒
            except subprocess.TimeoutExpired:
                print("[!] Process did not terminate gracefully, killing it.")
                process.kill()  # 强制杀死


if __name__ == "__main__":
    print("--- Starting Main Controller for Video Stream ---")

    # 从算法B获取实时视频流
    stream_generator = get_jpeg_stream_from_algorithm(
        PYTHON_B_PATH,
        SCRIPT_B_PATH,
        INPUT_VIDEO   # 或者用 '0' 来打开默认摄像头
        # '0'
    )

    frame_counter = 0
    save_every_n_frames = 30  # 每30帧保存一张图片以作演示

    print("[+]Receiving JPEG stream from Algorithm B...")

    for jpeg_frame in stream_generator:
        frame_counter += 1
        print(f"\r[*] Received frame {frame_counter}, size: {len(jpeg_frame)} bytes", end="")

    # 在环境A中，你可以对这个jpeg_frame做任何事
    # 例如：显示它、再次处理它、或者发送到别处

    # 为了演示，我们每隔一段时间保存一帧
        if frame_counter % save_every_n_frames == 0:
            output_filename = f"output_frame_{frame_counter}.jpg"
        with open(output_filename, "wb") as f:
            f.write(jpeg_frame)
        print(f" | Saved frame to {output_filename}")
        print("main controller terminated")