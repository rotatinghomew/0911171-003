"""按真实预测时点构造建模样本。

特征只使用 predict_as_of 之前发生且入账的活动；标签只使用 label_as_of
之前入账的结果。两个时点分开传入，回放历史时各自取当时的真实值。
"""

from .features import features_at
from .labels import training_label

_ID_KEYS = {"learner_id", "cohort", "started_at", "ended_at"}


def build_examples(enrollments: list[dict], activity: list[dict], outcomes: list[dict],
                   predict_as_of: int, label_as_of: int) -> list[dict]:
    activity_by_learner: dict[str, list[dict]] = {}
    for event in activity:
        activity_by_learner.setdefault(event["learner_id"], []).append(event)
    outcomes_by_learner: dict[str, list[dict]] = {}
    for record in outcomes:
        outcomes_by_learner.setdefault(record["learner_id"], []).append(record)
    examples = []
    for enrollment in sorted(enrollments, key=lambda row: row["learner_id"]):
        learner_id = enrollment["learner_id"]
        label, reason = training_label(enrollment, outcomes_by_learner.get(learner_id, []), label_as_of)
        examples.append({
            "learner_id": learner_id,
            "cohort": enrollment["cohort"],
            "predict_as_of": predict_as_of,
            "label_as_of": label_as_of,
            "features": features_at(enrollment, activity_by_learner.get(learner_id, []), predict_as_of),
            "label": label,
            "excluded": reason if label is None else None,
            "attributes": {key: value for key, value in enrollment.items() if key not in _ID_KEYS},
        })
    return examples
