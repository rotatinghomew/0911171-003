from .features import features_at
from .model import score


def review(enrollments: list[dict], events: list[dict], as_of: int, parameters: dict[str, float]) -> list[dict]:
    results = []
    for enrollment in enrollments:
        history = [event for event in events if event["learner_id"] == enrollment["learner_id"]]
        feature_set = features_at(enrollment, history, as_of)
        results.append({"learner_id": enrollment["learner_id"], "as_of": as_of, "features": feature_set, "score": score(feature_set, parameters)})
    return results
