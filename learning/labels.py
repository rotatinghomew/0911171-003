"""结果标签口径：退课、尚未结课与结课后修正分别处理，不一律记为失败。

- completed → 1（完成），failed → 0（未完成）；
- withdrawn（退课）不进入成败标签，单独排除；
- 尚未结课或结课后没有结果记录，一律视为未知而不是失败；
- 结课后修正以入账时间为准：label_as_of 之前入账的最新记录生效，
  之后入账的修正不影响该时点的历史标签。
"""

OUTCOME_COMPLETED = "completed"
OUTCOME_FAILED = "failed"
OUTCOME_WITHDRAWN = "withdrawn"

LABEL_COMPLETED = 1
LABEL_FAILED = 0

REASON_WITHDRAWN = "退课，不计入成败标签"
REASON_CENSORED = "尚未结课，结果未知"
REASON_NO_OUTCOME = "结课后缺少结果记录，视为未知"


def outcome_as_of(records: list[dict], as_of: int) -> dict | None:
    """取 as_of 之前入账的最新结果记录；没有则返回 None。"""
    visible = [record for record in records if record["recorded_at"] <= as_of]
    if not visible:
        return None
    return max(visible, key=lambda record: record["recorded_at"])


def training_label(enrollment: dict, records: list[dict], label_as_of: int) -> tuple[int | None, str]:
    """返回 (label, 说明)；label 为 None 表示该记录不进入训练或评估。"""
    record = outcome_as_of(records, label_as_of)
    if record is None:
        ended_at = enrollment.get("ended_at")
        if ended_at is None or ended_at > label_as_of:
            return None, REASON_CENSORED
        return None, REASON_NO_OUTCOME
    status = record["status"]
    if status == OUTCOME_COMPLETED:
        return LABEL_COMPLETED, "完成"
    if status == OUTCOME_FAILED:
        return LABEL_FAILED, "未完成"
    if status == OUTCOME_WITHDRAWN:
        return None, REASON_WITHDRAWN
    raise ValueError(f"未知结果状态: {status}")
