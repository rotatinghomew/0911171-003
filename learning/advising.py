from copy import deepcopy

from .review import review


class AdvisingDesk:
    def __init__(self, seats: dict[str, int]) -> None:
        if any(count < 0 for count in seats.values()):
            raise ValueError("辅导席位不能为负")
        self.seats = dict(seats)
        self.reservations: dict[str, list[dict]] = {}

    def propose(self, enrollments: list[dict], activity: list[dict], as_of: int, parameters: dict[str, float]) -> list[dict]:
        predictions = review(enrollments, activity, as_of, parameters)
        groups = {row["learner_id"]: row["cohort"] for row in enrollments}
        remaining = dict(self.seats)
        proposals = []
        for prediction in sorted(predictions, key=lambda row: (row["score"], row["learner_id"])):
            cohort = groups[prediction["learner_id"]]
            if remaining.get(cohort, 0) > 0:
                proposals.append({**prediction, "cohort": cohort})
                remaining[cohort] -= 1
        return proposals

    def reserve(self, session_id: str, proposals: list[dict]) -> list[dict]:
        if session_id in self.reservations:
            if self.reservations[session_id] != proposals:
                raise ValueError("已确认的辅导名单不能被同号覆盖")
            return deepcopy(self.reservations[session_id])
        identifiers = [row["learner_id"] for row in proposals]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("同一场辅导不能重复预留学员")
        for cohort, seats in self.seats.items():
            if sum(row["cohort"] == cohort for row in proposals) > seats:
                raise ValueError("辅导名单超过席位")
        if any(row["cohort"] not in self.seats for row in proposals):
            raise ValueError("辅导班次不存在")
        self.reservations[session_id] = deepcopy(proposals)
        return deepcopy(proposals)
