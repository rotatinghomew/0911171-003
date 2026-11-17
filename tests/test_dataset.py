import unittest

from learning.dataset import build_examples


class DatasetTests(unittest.TestCase):
    def test_backfilled_activity_does_not_leak_into_past_predictions(self):
        enrollment = {"learner_id": "a", "cohort": "c1", "started_at": 0, "ended_at": 1000}
        activity = [{"learner_id": "a", "occurred_at": 10, "recorded_at": 900, "minutes": 60}]
        outcomes = [{"learner_id": "a", "status": "completed", "recorded_at": 950}]
        past = build_examples([enrollment], activity, outcomes, predict_as_of=100, label_as_of=950)[0]
        self.assertEqual(past["features"]["visible_minutes"], 0)
        later = build_examples([enrollment], activity, outcomes, predict_as_of=1000, label_as_of=950)[0]
        self.assertEqual(later["features"]["visible_minutes"], 60)

    def test_labels_come_from_label_time_not_predict_time(self):
        enrollment = {"learner_id": "a", "cohort": "c1", "started_at": 0, "ended_at": 500}
        outcomes = [{"learner_id": "a", "status": "failed", "recorded_at": 600}]
        early = build_examples([enrollment], [], outcomes, predict_as_of=100, label_as_of=550)[0]
        self.assertIsNone(early["label"])
        late = build_examples([enrollment], [], outcomes, predict_as_of=100, label_as_of=650)[0]
        self.assertEqual(late["label"], 0)

    def test_attributes_are_carried_for_group_analysis(self):
        enrollment = {"learner_id": "a", "cohort": "c1", "started_at": 0, "ended_at": 10, "channel": "partner"}
        example = build_examples([enrollment], [], [], predict_as_of=5, label_as_of=5)[0]
        self.assertEqual(example["attributes"], {"channel": "partner"})
        self.assertEqual(example["cohort"], "c1")


if __name__ == "__main__":
    unittest.main()
