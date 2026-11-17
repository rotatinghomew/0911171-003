import unittest

from learning.labels import (
    REASON_CENSORED,
    REASON_NO_OUTCOME,
    REASON_WITHDRAWN,
    outcome_as_of,
    training_label,
)


class LabelTests(unittest.TestCase):
    def test_withdrawal_is_excluded_not_failure(self):
        enrollment = {"started_at": 0, "ended_at": 100}
        records = [{"learner_id": "a", "status": "withdrawn", "recorded_at": 90}]
        label, reason = training_label(enrollment, records, 200)
        self.assertIsNone(label)
        self.assertEqual(reason, REASON_WITHDRAWN)

    def test_unfinished_course_is_censored_not_failure(self):
        label, reason = training_label({"started_at": 0, "ended_at": None}, [], 50)
        self.assertIsNone(label)
        self.assertEqual(reason, REASON_CENSORED)
        label, reason = training_label({"started_at": 0, "ended_at": 500}, [], 50)
        self.assertIsNone(label)
        self.assertEqual(reason, REASON_CENSORED)

    def test_missing_outcome_after_end_is_unknown_not_failure(self):
        label, reason = training_label({"started_at": 0, "ended_at": 100}, [], 200)
        self.assertIsNone(label)
        self.assertEqual(reason, REASON_NO_OUTCOME)

    def test_correction_after_label_time_does_not_rewrite_history(self):
        records = [
            {"learner_id": "a", "status": "failed", "recorded_at": 90},
            {"learner_id": "a", "status": "completed", "recorded_at": 150, "corrects": True},
        ]
        enrollment = {"started_at": 0, "ended_at": 80}
        self.assertEqual(training_label(enrollment, records, 120)[0], 0)
        self.assertEqual(training_label(enrollment, records, 200)[0], 1)

    def test_outcome_as_of_picks_latest_visible_record(self):
        records = [
            {"learner_id": "a", "status": "completed", "recorded_at": 100},
            {"learner_id": "a", "status": "failed", "recorded_at": 80},
        ]
        self.assertEqual(outcome_as_of(records, 90)["status"], "failed")
        self.assertEqual(outcome_as_of(records, 100)["status"], "completed")
        self.assertIsNone(outcome_as_of(records, 50))


if __name__ == "__main__":
    unittest.main()
