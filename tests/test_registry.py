import tempfile
import unittest
from pathlib import Path

from learning import sample_data
from learning.pipeline import run_propose, run_train
from learning.registry import ModelRegistry
from learning.reservations import ReservationStore


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.artifacts = Path(self.tmp.name) / "artifacts"
        self.data = sample_data.generate(n_cohorts=4, cohort_size=24)

    def tearDown(self):
        self.tmp.cleanup()

    def test_retraining_same_data_yields_same_candidate_version(self):
        first = run_train(self.data, self.artifacts)
        second = run_train(self.data, self.artifacts)
        self.assertEqual(first["versions"], second["versions"])
        self.assertEqual(len(ModelRegistry(self.artifacts).candidates()), 2)

    def test_retraining_does_not_move_champion_or_confirmed_lists(self):
        versions = run_train(self.data, self.artifacts)["versions"]
        registry = ModelRegistry(self.artifacts)
        registry.promote(versions["logistic_baseline"])
        cohort = "batch-4"
        payload = run_propose(self.data, self.artifacts, cohort, session="week1", reservations_dir=Path(self.tmp.name) / "res")
        confirmed = payload["reserved"]

        changed = sample_data.generate(n_cohorts=5, cohort_size=24)
        new_versions = run_train(changed, self.artifacts)["versions"]
        self.assertNotEqual(new_versions["logistic_baseline"], versions["logistic_baseline"])
        self.assertEqual(registry.champion_version(), versions["logistic_baseline"])
        self.assertEqual(ReservationStore(Path(self.tmp.name) / "res").get("week1"), confirmed)

        replayed = run_propose(self.data, self.artifacts, cohort, session="week1", reservations_dir=Path(self.tmp.name) / "res")
        self.assertEqual(replayed["reserved"], confirmed)

    def test_promote_requires_existing_candidate(self):
        with self.assertRaises(ValueError):
            ModelRegistry(self.artifacts).promote("v_missing")

    def test_propose_requires_promoted_model(self):
        run_train(self.data, self.artifacts)
        with self.assertRaises(RuntimeError):
            run_propose(self.data, self.artifacts, "batch-4")

    def test_confirmed_session_cannot_be_overwritten_by_another_list(self):
        store = ReservationStore(Path(self.tmp.name) / "res")
        store.reserve("week1", [{"learner_id": "a", "cohort": "c"}])
        with self.assertRaises(ValueError):
            store.reserve("week1", [{"learner_id": "b", "cohort": "c"}])
        self.assertEqual(store.get("week1"), [{"learner_id": "a", "cohort": "c"}])


if __name__ == "__main__":
    unittest.main()
