def features_at(enrollment: dict, activity: list[dict], as_of: int) -> dict[str, float | int]:
    visible = [item for item in activity if item["recorded_at"] <= as_of and item["occurred_at"] <= as_of]
    return {
        "days_since_start": max(0, (as_of - enrollment["started_at"]) // 86400),
        "visible_sessions": len(visible),
        "visible_minutes": sum(int(item["minutes"]) for item in visible),
    }
