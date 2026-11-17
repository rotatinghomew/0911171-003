"""预测时点特征：只使用 as_of 之前发生且入账的活动。

补录的活动（occurred_at 早、recorded_at 晚）在 as_of 之前不可见，
不会泄露到过去的预测里。所有时间为 UTC 秒。
"""

DAY = 86400

FEATURE_NAMES = (
    "days_since_start",
    "visible_sessions",
    "visible_minutes",
    "active_days",
    "minutes_last_7d",
    "days_since_last_activity",
    "avg_minutes_per_session",
)


def features_at(enrollment: dict, activity: list[dict], as_of: int) -> dict[str, float | int]:
    visible = [item for item in activity if item["recorded_at"] <= as_of and item["occurred_at"] <= as_of]
    minutes = [int(item["minutes"]) for item in visible]
    occurred = [int(item["occurred_at"]) for item in visible]
    days_since_start = max(0, (as_of - enrollment["started_at"]) // DAY)
    return {
        "days_since_start": days_since_start,
        "visible_sessions": len(visible),
        "visible_minutes": sum(minutes),
        "active_days": len({ts // DAY for ts in occurred}),
        "minutes_last_7d": sum(m for m, ts in zip(minutes, occurred) if ts > as_of - 7 * DAY),
        "days_since_last_activity": days_since_start if not occurred else max(0, (as_of - max(occurred)) // DAY),
        "avg_minutes_per_session": (sum(minutes) / len(minutes)) if minutes else 0.0,
    }
