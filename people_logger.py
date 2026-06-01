import os
import cv2
import csv
import time
import numpy as np
import onnxruntime as ort
from datetime import datetime


RETENTION_DAYS = 7
last_cleanup_date = None

def cleanup_old_images():
    cutoff = time.time() - RETENTION_DAYS*24*60*60
    removed = 0

    if not os.path.exists(IMAGE_DIR):
        return


    for filename in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR,filename)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                removed +=1
            except Exception as e:
                print(f"删除失败:{path} -> {e}")
    print(f"清理完成，删除了 {removed}张7天前的照片")


# ==========================================
# 配置
# ==========================================


RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/sub/av_stream"

MODEL_PATH = "/home/pi/jtz/yolo_v6/models/yolov6n.onnx"

LOG_DIR = "/home/pi/jtz/yolo_v6/logs"

IMAGE_DIR = "/home/pi/jtz/yolo_v6/logs/images"

CSV_PATH = "/home/pi/jtz/yolo_v6/logs/people_log.csv"

# 置信度阈值
CONF_THRES = 0.4

# NMS 去重阈值
NMS_THRES = 0.45

# 检测间隔（秒）
SAMPLE_INTERVAL = 120

# 输入尺寸
INPUT_SIZE = 640

# ==========================================
# 初始化
# ==========================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print("加载 ONNX 模型...")

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

print("模型加载成功")

# 初始化 CSV
if not os.path.exists(CSV_PATH):

    with open(CSV_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "people_count",
            "image_path"
        ])

print("日志系统初始化完成")

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

        # 打开 RTSP
        cap = cv2.VideoCapture(RTSP_URL)

        if not cap.isOpened():

            print("RTSP 打开失败")

            time.sleep(SAMPLE_INTERVAL)

            continue

        ret, frame = cap.read()

        cap.release()

        if not ret:

            print("读取视频帧失败")

            time.sleep(SAMPLE_INTERVAL)

            continue

        orig_h, orig_w = frame.shape[:2]

        # ==========================================
        # 预处理
        # ==========================================

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, axis=0)

        # ==========================================
        # 推理
        # ==========================================

        outputs = session.run(None, {input_name: img})

        preds = outputs[0][0]

        preds = preds.T

        boxes = []

        scores = []

        # ==========================================
        # 收集检测框
        # ==========================================

        for det in preds:

            x, y, w, h, score = det

            if score < CONF_THRES:
                continue

            x1 = int((x - w / 2) * orig_w / INPUT_SIZE)

            y1 = int((y - h / 2) * orig_h / INPUT_SIZE)

            x2 = int((x + w / 2) * orig_w / INPUT_SIZE)

            y2 = int((y + h / 2) * orig_h / INPUT_SIZE)

            boxes.append([x1, y1, x2 - x1, y2 - y1])

            scores.append(float(score))

        # ==========================================
        # NMS 去重
        # ==========================================

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF_THRES,
            NMS_THRES
        )

        people_count = 0

        if len(indices) > 0:

            for idx in indices.flatten():

                people_count += 1

                x, y, w, h = boxes[idx]

                score = scores[idx]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

        # ==========================================
        # 显示人数
        # ==========================================

        cv2.putText(
            frame,
            f"People: {people_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        # ==========================================
        # 保存图片
        # ==========================================

        cv2.imwrite(image_path, frame)

        # ==========================================
        # 写 CSV
        # ==========================================

        with open(CSV_PATH, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                timestamp,
                people_count,
                image_path
            ])

        print(f"人数统计: {people_count}")

        print(f"图片保存: {image_path}")

        print("CSV 日志已写入")

    except Exception as e:

        print("程序异常:", e)

    # 等待下一次检测
    time.sleep(SAMPLE_INTERVAL)
