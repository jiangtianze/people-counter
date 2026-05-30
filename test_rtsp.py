import cv2

RTSP_URL = "rtsp://admin:zaq1xsw2@192.168.3.220:554/h264/ch1/sub/av_stream"

print("正在连接 RTSP...")

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP 打开失败")
    exit()

print("RTSP 连接成功")

ret, frame = cap.read()

if not ret:
    print("读取视频帧失败")
else:
    print("成功读取一帧")

    cv2.imwrite("test.jpg", frame)
    print("截图已保存为 test.jpg")

cap.release()
