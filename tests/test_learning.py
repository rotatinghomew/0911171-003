import unittest

from learning.features import features_at
from learning.review import review


class LearningTests(unittest.TestCase):
    def test_future_activity_is_not_visible(self):
        rows = [{"recorded_at": 20, "occurred_at": 10, "minutes": 8}, {"recorded_at": 10, "occurred_at": 30, "minutes": 9}]
        self.assertEqual(features_at({"started_at": 0}, rows, 15)["visible_sessions"], 0)

    def test_review_keeps_prediction_time(self):
        result = review([{"learner_id": "a", "started_at": 0}], [], 100, {})
        self.assertEqual(result[0]["as_of"], 100)


if __name__ == "__main__":
    unittest.main()
