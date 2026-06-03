from datetime import datetime
from flask import Flask, render_template,send_file,jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route("/latest_image")
def latest_image():
    df = pd.read_csv(CSV_PATH)

    if len(df) == 0:
        return "No Image"

    image_path = df.iloc[-1, 2]

    return send_file(image_path)

@app.route("/api/status")
def api_status():

    df = pd.read_csv(CSV_PATH)

    if len(df) == 0:
        return jsonify({
            "count": 0,
            "time": "",
            "status": "离线"
        })

    current_count = int(df.iloc[-1,1])
    latest_time = str(df.iloc[-1,0])

    return jsonify({
        "count": current_count,
        "time": latest_time,
        "status": "在线"
    })

@app.route("/api/chart")
def api_chart():

    df = pd.read_csv(CSV_PATH)

    times = df["timestamp"].tail(50).tolist()
    counts = df["people_count"].tail(50).tolist()

    return jsonify({
        "times": times,
        "counts": counts
    })

CSV_PATH = os.path.expanduser("~/jtz/yolo_v6/logs/people_log.csv")

@app.route("/")
def index():
    df = pd.read_csv(CSV_PATH)

    times = df["timestamp"].tail(50).tolist()
    counts = df["people_count"].tail(50).tolist()

    current_count = counts[-1] if counts else 0

    latest_time = times[-1] if times else "暂无数据"
 
    status = "离线"

    if times:
        last_update = datetime.strptime(
            latest_time,
            "%Y-%m-%d %H:%M:%S"
        )

        seconds = (
            datetime.now() - last_update
        ).total_seconds()

        if seconds < 300:
            status = "在线"

    max_count = max(counts) if counts else 0

    avg_count = round(sum(counts) / len(counts), 1) if counts else 0

    latest_image = ""

    if len(df)>0:
        latest_image = df.iloc[-1,2]

    recent_records = []

    for i in range(len(df) - 1, max(len(df) - 11, -1), -1):
        recent_records.append({
            "time": str(df.iloc[i,0]),
            "count": int(df.iloc[i,1])
        })
    return render_template(
        "index.html",
        times=times,
        counts=counts,
        current_count=current_count,
        latest_image=latest_image,
        latest_time=latest_time,
        max_count=max_count,
        avg_count=avg_count,
        recent_records=recent_records,
        status=status

    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
