"""测试主码流连通性（与 people_logger 相同的 ffmpeg 轻量化抓帧）。"""

import os
import subprocess
import cv2

RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/main/av_stream"
RTSP_TRANSPORT = "tcp"
SAVE_MAX_WIDTH = 1280
OUTPUT = "test_main.jpg"


def capture_main_stream():
    cmd = [
        "ffmpeg", "-y", "-loglevel", "info",
        "-rtsp_transport", RTSP_TRANSPORT,
        "-i", RTSP_URL,
        "-frames:v", "1",
        "-vf", f"scale='min({SAVE_MAX_WIDTH},iw)':-1",
        "-q:v", "5",
        OUTPUT,
    ]
    print("正在连接主码流 RTSP...")
    print("URL:", RTSP_URL)
    result = subprocess.run(cmd, timeout=30)
    if result.returncode != 0:
        print("ffmpeg 抓帧失败，请检查主码流地址与网络")
        return False

    frame = cv2.imread(OUTPUT)
    if frame is None:
        print("图片读取失败")
        return False

    h, w = frame.shape[:2]
    print(f"成功！已保存 {OUTPUT}，尺寸 {w}x{h}")
    return True


if __name__ == "__main__":
    if not capture_main_stream():
        raise SystemExit(1)
