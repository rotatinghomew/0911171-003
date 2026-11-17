import unittest

from learning.advising import AdvisingDesk
from learning.features import FEATURE_NAMES
from learning.models import BASELINE, TrainedModel


def _examples():
    rows = []
    for i in range(20):
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["visible_minutes"] = float(i * 30)
        rows.append({"features": features, "label": 1 if i >= 10 else 0})
    return rows


class AdvisingModelTests(unittest.TestCase):
    def test_propose_with_model_orders_by_score_and_marks_version(self):
        model = TrainedModel.fit(BASELINE, _examples())
        model.version = "v_test"
        desk = AdvisingDesk({"autumn": 2})
        learners = [
            {"learner_id": "busy", "cohort": "autumn", "started_at": 0},
            {"learner_id": "quiet", "cohort": "autumn", "started_at": 0},
            {"learner_id": "mid", "cohort": "autumn", "started_at": 0},
        ]
        events = [
            {"learner_id": "busy", "recorded_at": 1, "occurred_at": 1, "minutes": 300},
            {"learner_id": "mid", "recorded_at": 1, "occurred_at": 1, "minutes": 120},
        ]
        proposal = desk.propose(learners, events, 2 * 86400, model=model)
        self.assertEqual(proposal[0]["learner_id"], "quiet")
        self.assertEqual(len(proposal), 2)
        self.assertTrue(all(row["model_version"] == "v_test" for row in proposal))
        scores = [row["score"] for row in proposal]
        self.assertEqual(scores, sorted(scores))

    def test_propose_requires_model_or_parameters(self):
        desk = AdvisingDesk({"autumn": 1})
        with self.assertRaises(ValueError):
            desk.propose([{"learner_id": "a", "cohort": "autumn", "started_at": 0}], [], 10)

    def test_retraining_does_not_change_confirmed_reservation(self):
        desk = AdvisingDesk({"autumn": 1})
        learners = [{"learner_id": "a", "cohort": "autumn", "started_at": 0}]
        first = desk.propose(learners, [], 10, {"intercept": -1})
        desk.reserve("week1", first)
        # 重新训练/换参数后的新名单不能覆盖已确认场次
        second = desk.propose(learners, [], 10, {"intercept": 5})
        self.assertNotEqual(first[0]["score"], second[0]["score"])
        with self.assertRaises(ValueError):
            desk.reserve("week1", second)
        self.assertEqual(desk.reservations["week1"], first)


if __name__ == "__main__":
    unittest.main()
