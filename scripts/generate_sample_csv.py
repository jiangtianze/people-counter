"""生成模拟历史 CSV，用于本地验证预测模块。"""

import csv
import os
import random
from datetime import datetime, timedelta

OUTPUT = os.path.join(os.path.dirname(__file__), "sample_data", "people_log.csv")

# 工作日 / 周末各时段基准人数（体现课表与作息规律）
WEEKDAY_BASE = {
    8: 5, 9: 18, 10: 22, 11: 20, 12: 15, 13: 12,
    14: 20, 15: 21, 16: 19, 17: 14, 18: 8, 19: 5, 20: 3,
}
WEEKEND_BASE = {
    9: 2, 10: 4, 11: 5, 12: 4, 13: 3, 14: 4,
    15: 5, 16: 4, 17: 3, 18: 2,
}


def base_count(dt: datetime) -> int:
    hour = dt.hour
    if dt.weekday() >= 5:
        base = WEEKEND_BASE.get(hour, 1)
    else:
        base = WEEKDAY_BASE.get(hour, 2)
    return max(0, base + random.randint(-2, 2))


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()

    rows = []
    current = start
    while current <= end:
        if 8 <= current.hour <= 20:
            count = base_count(current)
            # 模拟偶发异常（中位数应能抗住）
            if random.random() < 0.01:
                count = random.randint(35, 45)
            rows.append([
                current.strftime("%Y-%m-%d %H:%M:%S"),
                count,
                f"/tmp/sample_{current.strftime('%Y%m%d_%H%M%S')}.jpg",
            ])
        current += timedelta(minutes=2)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "people_count", "image_path"])
        writer.writerows(rows)

    print(f"已生成 {len(rows)} 条记录 -> {OUTPUT}")


if __name__ == "__main__":
    main()
