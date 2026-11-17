import unittest

from learning.features import FEATURE_NAMES
from learning.models import BASELINE, CANDIDATE, TrainedModel


def _examples():
    rows = []
    for i in range(40):
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["visible_minutes"] = float(i * 10)
        features["visible_sessions"] = i % 5
        rows.append({"learner_id": f"s{i}", "features": features, "label": 1 if i >= 20 else 0})
    return rows


class ModelTests(unittest.TestCase):
    def test_fit_requires_both_classes(self):
        with self.assertRaises(ValueError):
            TrainedModel.fit(BASELINE, [{"features": {}, "label": 1}, {"features": {}, "label": 1}])

    def test_unlabeled_examples_are_not_used_for_training(self):
        rows = _examples() + [{"features": {}, "label": None}]
        model = TrainedModel.fit(BASELINE, rows)
        self.assertEqual(len(model.feature_names), len(FEATURE_NAMES))

    def test_training_is_deterministic(self):
        first = TrainedModel.fit(CANDIDATE, _examples())
        second = TrainedModel.fit(CANDIDATE, _examples())
        features = {name: 1.0 for name in FEATURE_NAMES}
        self.assertEqual(first.predict_proba_one(features), second.predict_proba_one(features))

    def test_baseline_exposes_interpretable_coefficients(self):
        coefficients = TrainedModel.fit(BASELINE, _examples()).coefficients()
        self.assertIn("intercept", coefficients)
        self.assertIn("visible_minutes", coefficients)
        self.assertGreater(coefficients["visible_minutes"], 0)

    def test_candidate_has_no_coefficients(self):
        self.assertIsNone(TrainedModel.fit(CANDIDATE, _examples()).coefficients())

    def test_scores_are_probabilities(self):
        model = TrainedModel.fit(BASELINE, _examples())
        for features in [{name: 0.0 for name in FEATURE_NAMES}, {name: 100.0 for name in FEATURE_NAMES}]:
            self.assertGreaterEqual(model.predict_proba_one(features), 0.0)
            self.assertLessEqual(model.predict_proba_one(features), 1.0)


if __name__ == "__main__":
    unittest.main()
