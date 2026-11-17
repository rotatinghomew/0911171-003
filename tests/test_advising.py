import unittest

from learning.advising import AdvisingDesk


class AdvisingTests(unittest.TestCase):
    def test_risk_order_and_seats_control_proposal(self):
        desk = AdvisingDesk({"autumn": 1})
        learners = [
            {"learner_id": "a", "cohort": "autumn", "started_at": 0},
            {"learner_id": "b", "cohort": "autumn", "started_at": 0},
        ]
        events = [{"learner_id": "a", "recorded_at": 1, "occurred_at": 1, "minutes": 60}]
        proposal = desk.propose(learners, events, 2, {"visible_minutes": 0.1})
        self.assertEqual([row["learner_id"] for row in proposal], ["b"])
        self.assertEqual(desk.reserve("week1", proposal), proposal)
        self.assertEqual(desk.reserve("week1", proposal), proposal)

    def test_confirmation_cannot_replace_a_reserved_session(self):
        desk = AdvisingDesk({"autumn": 1})
        desk.reserve("week1", [{"learner_id": "a", "cohort": "autumn"}])
        with self.assertRaises(ValueError):
            desk.reserve("week1", [{"learner_id": "b", "cohort": "autumn"}])
