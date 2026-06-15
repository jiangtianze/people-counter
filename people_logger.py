import os
import cv2
import csv
import time
import subprocess
import numpy as np
import onnxruntime as ort
from datetime import datetime


RETENTION_DAYS = 7
last_cleanup_date = None

# ==========================================
# 配置
# ==========================================

# 主码流（检测精度更高；通过下方轻量化策略降低 Pi 3B 负载）
RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/main/av_stream"
RTSP_TRANSPORT = "tcp"

MODEL_PATH = "/home/pi/jtz/yolo_v6/models/yolov6n.onnx"

LOG_DIR = "/home/pi/jtz/yolo_v6/logs"
IMAGE_DIR = "/home/pi/jtz/yolo_v6/logs/images"
CSV_PATH = "/home/pi/jtz/yolo_v6/logs/people_log.csv"

CONF_THRES = 0.4
NMS_THRES = 0.45

# 主码流建议 180~300 秒；若仍卡顿可再加大
SAMPLE_INTERVAL = 180

INPUT_SIZE = 640
# 存图/画框最大宽度，避免保存 1080p 原图占满 CPU 与 SD 卡
SAVE_MAX_WIDTH = 1280

# Pi 3B 优先 ffmpeg 单帧抓取（解码后立即缩放，内存占用更小）
USE_FFMPEG_CAPTURE = True
FFMPEG_TIMEOUT = 30

# OpenCV 备用方案：丢弃缓冲中的旧帧后再取最新一帧
RTSP_FLUSH_GRABS = 3

# ONNX 线程数（Pi 3B 四核，留余量给系统与 SSH）
ONNX_THREADS = 2


def cleanup_old_images():
    cutoff = time.time() - RETENTION_DAYS * 24 * 60 * 60
    removed = 0

    if not os.path.exists(IMAGE_DIR):
        return

    for filename in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                removed += 1
            except Exception as e:
                print(f"删除失败:{path} -> {e}")
    print(f"清理完成，删除了 {removed} 张 7 天前的照片")


def limit_frame_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def capture_frame_ffmpeg(url):
    """ffmpeg 只抓 1 帧并在解码时缩放，适合 Pi 3B + 主码流。"""
    tmp_path = os.path.join(IMAGE_DIR, "_tmp_capture.jpg")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-rtsp_transport", RTSP_TRANSPORT,
        "-i", url,
        "-frames:v", "1",
        "-vf", f"scale='min({SAVE_MAX_WIDTH},iw)':-1",
        "-q:v", "5",
        tmp_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            timeout=FFMPEG_TIMEOUT,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print("ffmpeg 抓帧超时")
        return None

    if result.returncode != 0:
        err = (result.stderr or "").strip()
        print(f"ffmpeg 抓帧失败: {err or result.returncode}")
        return None

    frame = cv2.imread(tmp_path)
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if frame is None:
        print("ffmpeg 输出图片读取失败")
    return frame


def capture_frame_opencv(url):
    """OpenCV 备用：限制缓冲、flush 旧帧、立即缩小。"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("OpenCV RTSP 打开失败")
        return None

    frame = None
    try:
        for _ in range(RTSP_FLUSH_GRABS):
            if not cap.grab():
                break

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            ret, frame = cap.read()
    finally:
        cap.release()

    if frame is None:
        print("OpenCV 读取视频帧失败")
        return None

    return limit_frame_width(frame, SAVE_MAX_WIDTH)


def capture_frame(url):
    if USE_FFMPEG_CAPTURE:
        frame = capture_frame_ffmpeg(url)
        if frame is not None:
            return frame
        print("ffmpeg 失败，尝试 OpenCV 备用抓帧...")

    return capture_frame_opencv(url)


def preprocess_for_inference(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def detect_people(frame, session, input_name):
    work_h, work_w = frame.shape[:2]

    outputs = session.run(None, {input_name: preprocess_for_inference(frame)})
    preds = outputs[0][0].T

    boxes = []
    scores = []

    for det in preds:
        x, y, w, h, score = det
        if score < CONF_THRES:
            continue

        x1 = int((x - w / 2) * work_w / INPUT_SIZE)
        y1 = int((y - h / 2) * work_h / INPUT_SIZE)
        bw = int(w * work_w / INPUT_SIZE)
        bh = int(h * work_h / INPUT_SIZE)

        boxes.append([x1, y1, bw, bh])
        scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES)

    annotated = frame.copy()
    people_count = 0

    if len(indices) > 0:
        for idx in indices.flatten():
            people_count += 1
            x, y, w, h = boxes[idx]
            score = scores[idx]

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.putText(
        annotated,
        f"People: {people_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    return people_count, annotated


# ==========================================
# 初始化
# ==========================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print("加载 ONNX 模型...")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = ONNX_THREADS
sess_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
print("模型加载成功")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "people_count", "image_path"])

print("日志系统初始化完成")
print(f"RTSP: {RTSP_URL}")
print(f"抓帧: {'ffmpeg' if USE_FFMPEG_CAPTURE else 'OpenCV'}, 存图最大宽度: {SAVE_MAX_WIDTH}")

# ==========================================
# 主循环
# ==========================================

while True:
    today = datetime.now().date()

    if last_cleanup_date != today:
        cleanup_old_images()
        last_cleanup_date = today

    try:
        print("=" * 60)

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        image_name = now.strftime("%Y-%m-%d_%H-%M-%S.jpg")
        image_path = os.path.join(IMAGE_DIR, image_name)

        print(f"[{timestamp}] 开始检测")

        frame = capture_frame(RTSP_URL)
        if frame is None:
            time.sleep(SAMPLE_INTERVAL)
            continue

        people_count, annotated = detect_people(frame, session, input_name)

        cv2.imwrite(image_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, people_count, image_path])

        print(f"人数统计: {people_count}")
        print(f"图片保存: {image_path} ({annotated.shape[1]}x{annotated.shape[0]})")
        print("CSV 日志已写入")

        del frame, annotated

    except Exception as e:
        print("程序异常:", e)

    time.sleep(SAMPLE_INTERVAL)
