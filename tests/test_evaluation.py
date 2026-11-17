import unittest

from learning import sample_data
from learning.evaluation import ReplayConfig, replay


def _report(**overrides):
    data = sample_data.generate(n_cohorts=4, cohort_size=24)
    config = ReplayConfig(seats=data["seats"], bootstrap=50, min_fold_n=10, min_fold_failures=3, min_group_failures=3, **overrides)
    return data, replay(data["enrollments"], data["activity"], data["outcomes"], config=config, eval_label_as_of=data["now"])


class EvaluationTests(unittest.TestCase):
    def test_training_only_uses_earlier_batches(self):
        _, report = _report()
        order = {batch["cohort"]: batch["start"] for batch in report["batches"]}
        for model in report["models"].values():
            for fold in model["folds"]:
                self.assertTrue(all(order[train] < order[fold["cohort"]] for train in fold["train_cohorts"]))

    def test_ongoing_batch_is_censored_not_counted_as_failure(self):
        _, report = _report()
        last = report["batches"][-1]["cohort"]
        for model in report["models"].values():
            fold = next(row for row in model["folds"] if row["cohort"] == last)
            self.assertEqual(fold["n_labeled"], 0)
            self.assertIn("尚未结课，结果未知", fold["excluded"])
            self.assertIsNone(fold["auc"])

    def test_selection_respects_seats_and_names_learners(self):
        data, report = _report()
        for model in report["models"].values():
            for fold in model["selection"]["folds"]:
                self.assertLessEqual(len(fold["selected"]), data["seats"][fold["cohort"]])
                self.assertTrue(all(row["learner_id"] for row in fold["selected"]))

    def test_report_covers_discrimination_calibration_and_groups(self):
        _, report = _report()
        for model in report["models"].values():
            pooled = model["pooled"]
            self.assertIsNotNone(pooled["auc"])
            self.assertIsNotNone(pooled["brier"])
            self.assertEqual(len(pooled["auc_ci95"]), 2)
            self.assertTrue(model["calibration"])
            self.assertEqual(sum(row["n"] for row in model["calibration"]), pooled["n"])
            self.assertTrue(any(key.startswith("channel=") for key in model["selection"]["groups"]))
            for group in model["selection"]["groups"].values():
                self.assertEqual(group["missed"], group["failures"] - group["captured"])

    def test_tiny_sample_is_flagged_not_hidden(self):
        data = sample_data.generate(n_cohorts=3, cohort_size=8)
        config = ReplayConfig(seats=2, bootstrap=10)
        report = replay(data["enrollments"], data["activity"], data["outcomes"], config=config, eval_label_as_of=data["now"])
        self.assertTrue(report["warnings"])

    def test_replay_is_reproducible(self):
        first_data, first = _report()
        _, second = _report()
        self.assertEqual(first["models"]["logistic_baseline"]["pooled"], second["models"]["logistic_baseline"]["pooled"])
        self.assertEqual(
            first["models"]["gbdt_candidate"]["selection"]["folds"],
            second["models"]["gbdt_candidate"]["selection"]["folds"],
        )
        self.assertEqual(first_data["now"], second["eval_label_as_of"])


if __name__ == "__main__":
    unittest.main()
