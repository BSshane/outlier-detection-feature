import cv2
import sys
import argparse
import numpy as np


def stream_processor(rtmp_url):
    """
    连接到RTMP流，处理每一帧，并将处理后的JPEG帧写入标准输出。
    """
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        # 将错误信息打印到 stderr，Flask可以捕获
        print(f"Error: Cannot open RTMP stream at {rtmp_url}", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # 如果视频流结束或读取失败，则退出循环
                print("Stream ended or failed to read frame.", file=sys.stderr)
                break

            # ==========================================================
            # === 在这里进行你的核心处理，例如人脸识别、异常检测等 ===
            # 示例：我们在这里画一个矩形和添加一些文字
            h, w, _ = frame.shape
            cv2.rectangle(frame, (50, 50), (w - 50, h - 50), (0, 255, 0), 2)
            timestamp = cv2.getTickCount() / cv2.getTickFrequency()
            cv2.putText(frame, f"Processed Time: {timestamp:.2f}s", (60, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # ==========================================================

            # 将处理后的帧编码为 JPEG 格式
            # quality 参数可以调整，值越低，图像越小，传输越快
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            result, encimg = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                continue

            # !!! 关键步骤：将JPEG二进制数据写入标准输出 !!!
            # 我们使用 sys.stdout.buffer 来写入二进制数据
            jpeg_bytes = encimg.tobytes()

            # 为了让Flask知道每一帧的边界，我们发送一个简单的分界符
            # HTTP multipart/x-mixed-replace 已经有自己的边界，
            # 但我们这里为了让Flask读取逻辑更简单，也加一个。
            # 或者更简单的方式是，直接发送数据流，由Flask处理。
            # 这里我们采用最简单的方式：直接输出JPEG数据，由Flask负责封装

            # 我们需要一种方式告诉Flask这一帧有多大。
            # 协议：先发送4个字节的长度头，再发送JPEG数据。
            length = len(jpeg_bytes).to_bytes(4, 'big')  # 4字节，大端字节序
            sys.stdout.buffer.write(length)
            sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()  # 非常重要！立即清空缓冲区，将数据发送出去

    except BrokenPipeError:
        # 当Flask端关闭读取管道时，这里会触发此异常，是正常退出
        print("Client disconnected, stopping stream.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        cap.release()
        print("Stream processor shut down.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RTMP stream and output JPEG frames to stdout.")
    parser.add_argument("--rtmp-url", required=True, help="The input RTMP stream URL.")
    args = parser.parse_args()
    stream_processor(args.rtmp_url)
