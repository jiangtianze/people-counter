import cv2
import numpy as np
import onnxruntime as ort

RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/sub/av_stream"

MODEL_PATH = "/home/pi/jtz/yolo_v6/models/yolov6n.onnx"

CONF_THRES = 0.5

# 加载模型
print("加载模型...")
session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

# 打开 RTSP
print("连接 RTSP...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP 打开失败")
    exit()

print("RTSP 连接成功")

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取失败")
        continue

    orig_h, orig_w = frame.shape[:2]

    # resize
    img = cv2.resize(frame, (640, 640))

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # float32
    img_input = img_rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    img_input = np.transpose(img_input, (2, 0, 1))

    # batch
    img_input = np.expand_dims(img_input, axis=0)

    # 推理
    outputs = session.run(None, {input_name: img_input})

    preds = outputs[0][0]   # (5,8400)

    preds = preds.T         # (8400,5)

    people_count = 0

    for det in preds:

        x, y, w, h, score = det

        if score < CONF_THRES:
            continue

        people_count += 1

        # 转回原图坐标
        x1 = int((x - w / 2) * orig_w / 640)
        y1 = int((y - h / 2) * orig_h / 640)
        x2 = int((x + w / 2) * orig_w / 640)
        y2 = int((y + h / 2) * orig_h / 640)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f"{score:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    cv2.putText(
        frame,
        f"People: {people_count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("People Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
