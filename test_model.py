import cv2
import numpy as np
import onnxruntime as ort

RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/sub/av_stream"

MODEL_PATH = "/home/pi/yolo_v6/models/yolov6n.onnx"

print("加载模型...")
session = ort.InferenceSession(MODEL_PATH)

print("打开 RTSP...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP 打开失败")
    exit()

ret, frame = cap.read()

if not ret:
    print("读取帧失败")
    exit()

print("读取视频成功")

# resize 到 YOLO 输入尺寸
img = cv2.resize(frame, (640, 640))

# BGR -> RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转 float32
img = img.astype(np.float32) / 255.0

# HWC -> CHW
img = np.transpose(img, (2, 0, 1))

# 增加 batch 维度
img = np.expand_dims(img, axis=0)

input_name = session.get_inputs()[0].name

print("开始推理...")
outputs = session.run(None, {input_name: img})

print("推理成功")
print("输出数量:", len(outputs))

for i, out in enumerate(outputs):
    print(f"输出 {i} shape:", out.shape)

cap.release()
