"""
基于长期历史规律的人数预测模块。

思路：将历史记录按「星期几 + 小时」聚合，用中位数刻画典型人流，
预测未来 12 小时时查表即可。中位数对偶发异常不敏感，
全部历史样本等权参与，自然体现工作日/周末与各时段差异。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

FORECAST_HOURS = 12
MIN_WEEKDAY_SAMPLES = 3
MIN_WEEKEND_SAMPLES = 3
MIN_GLOBAL_SAMPLES = 5


def _parse_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["people_count"] = pd.to_numeric(df["people_count"], errors="coerce")
    df = df.dropna(subset=["timestamp", "people_count"])
    df["people_count"] = df["people_count"].clip(lower=0)
    return df


def _hourly_median(df: pd.DataFrame) -> pd.DataFrame:
    """将原始采样记录聚合为「每个自然小时一条」的中位数人数。"""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "people_count", "weekday", "hour", "is_weekend"])

    hourly = (
        df.set_index("timestamp")
        .resample("h")["people_count"]
        .median()
        .dropna()
        .reset_index()
    )
    hourly["weekday"] = hourly["timestamp"].dt.weekday
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["is_weekend"] = hourly["weekday"] >= 5
    return hourly


def _build_profiles(hourly: pd.DataFrame) -> dict[str, Any]:
    weekday_profile: dict[tuple[int, int], float] = {}
    weekday_pool: dict[int, float] = {}
    weekend_profile: dict[int, float] = {}
    global_profile: dict[int, float] = {}

    for (weekday, hour), group in hourly.groupby(["weekday", "hour"]):
        weekday_profile[(int(weekday), int(hour))] = float(group["people_count"].median())

    weekday_data = hourly[~hourly["is_weekend"]]
    if not weekday_data.empty:
        for hour, group in weekday_data.groupby("hour"):
            weekday_pool[int(hour)] = float(group["people_count"].median())

    weekend_data = hourly[hourly["is_weekend"]]
    if not weekend_data.empty:
        for hour, group in weekend_data.groupby("hour"):
            weekend_profile[int(hour)] = float(group["people_count"].median())

    for hour, group in hourly.groupby("hour"):
        global_profile[int(hour)] = float(group["people_count"].median())

    overall_median = float(hourly["people_count"].median()) if not hourly.empty else 0.0

    return {
        "weekday_profile": weekday_profile,
        "weekday_pool": weekday_pool,
        "weekend_profile": weekend_profile,
        "global_profile": global_profile,
        "overall_median": overall_median,
        "hourly_count": len(hourly),
    }


def _lookup_prediction(
    target: datetime,
    profiles: dict[str, Any],
    hourly: pd.DataFrame,
) -> tuple[float, str]:
    weekday = target.weekday()
    hour = target.hour
    is_weekend = weekday >= 5

    key = (weekday, hour)
    weekday_profile = profiles["weekday_profile"]

    if key in weekday_profile:
        slot = hourly[(hourly["weekday"] == weekday) & (hourly["hour"] == hour)]
        if len(slot) >= MIN_WEEKDAY_SAMPLES:
            return weekday_profile[key], "weekday_hour"

    pool = profiles["weekend_profile"] if is_weekend else profiles["weekday_pool"]
    min_samples = MIN_WEEKEND_SAMPLES if is_weekend else MIN_WEEKDAY_SAMPLES

    if hour in pool:
        if is_weekend:
            slot = hourly[(hourly["is_weekend"]) & (hourly["hour"] == hour)]
        else:
            slot = hourly[(~hourly["is_weekend"]) & (hourly["hour"] == hour)]
        if len(slot) >= min_samples:
            label = "weekend_hour" if is_weekend else "weekday_pool"
            return pool[hour], label

    global_profile = profiles["global_profile"]
    if hour in global_profile:
        slot = hourly[hourly["hour"] == hour]
        if len(slot) >= MIN_GLOBAL_SAMPLES:
            return global_profile[hour], "global_hour"

    return profiles["overall_median"], "overall_median"


def predict_next_hours(
    csv_path: str,
    hours: int = FORECAST_HOURS,
    start_time: datetime | None = None,
) -> dict[str, Any]:
    """
    读取 CSV，预测未来 hours 个小时（整点）的人数。

    返回 JSON 友好结构，供 Flask API 与前端图表使用。
    """
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return _empty_result("暂无历史数据，无法预测")

    df = _parse_csv(df)
    if df.empty:
        return _empty_result("历史数据为空，无法预测")

    hourly = _hourly_median(df)
    if hourly.empty:
        return _empty_result("有效历史数据不足，无法预测")

    profiles = _build_profiles(hourly)

    if start_time is None:
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        start_time = start_time.replace(minute=0, second=0, microsecond=0)

    times: list[str] = []
    counts: list[int] = []
    labels: list[str] = []
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    for i in range(hours):
        target = start_time + timedelta(hours=i)
        value, source = _lookup_prediction(target, profiles, hourly)
        times.append(target.strftime("%m-%d %H:00"))
        counts.append(max(0, int(round(value))))
        labels.append(f"{weekday_names[target.weekday()]} {target.strftime('%H:00')}")

    return {
        "ok": True,
        "message": "预测成功",
        "method": "historical_median_profile",
        "method_desc": "基于全部历史中「星期×小时」典型值（中位数）查表预测，抗异常、体现长期规律",
        "forecast_hours": hours,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "times": times,
        "labels": labels,
        "counts": counts,
        "meta": {
            "raw_records": len(df),
            "hourly_records": len(hourly),
            "history_start": hourly["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
            "history_end": hourly["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
        },
    }


def _empty_result(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "message": message,
        "method": "historical_median_profile",
        "method_desc": "",
        "forecast_hours": FORECAST_HOURS,
        "start_time": "",
        "times": [],
        "labels": [],
        "counts": [],
        "meta": {},
    }
