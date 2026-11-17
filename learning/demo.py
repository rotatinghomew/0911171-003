from .advising import AdvisingDesk


def main() -> None:
    desk = AdvisingDesk({"autumn": 1})
    learners = [{"learner_id": "s-1", "started_at": 0, "cohort": "autumn"}]
    events = [{"learner_id": "s-1", "recorded_at": 10, "occurred_at": 8, "minutes": 30}]
    proposal = desk.propose(learners, events, 20, {"intercept": -1, "visible_sessions": 0.4})
    print(desk.reserve("week1", proposal))


if __name__ == "__main__":
    main()
