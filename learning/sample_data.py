"""确定性的合成历史数据，仅用于本地回放验证，不代表真实学员。

包含：多个按时间推进的课程批次、迟到/补录的活动、退课、尚未结课的
最新批次，以及结课后修正的结果记录，用来检验标签口径与泄露控制。
"""

import numpy as np

DAY = 86400
SEED = 20261117

_CHANNELS = ("online", "onsite", "partner")
_CHANNEL_EFFECT = {"online": 0.0, "onsite": 0.4, "partner": -0.7}


def generate(seed: int = SEED, n_cohorts: int = 6, cohort_size: int = 48, lead_days: int = 14) -> dict:
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000
    duration = 40 * DAY
    step = 45 * DAY
    enrollments, activity, outcomes = [], [], []
    for c in range(n_cohorts):
        cohort = f"batch-{c + 1}"
        start = t0 + c * step
        ongoing = c == n_cohorts - 1
        mix = np.array([0.5, 0.3, 0.2]) + c * np.array([-0.03, 0.0, 0.03])
        mix = np.clip(mix, 0.05, None)
        mix = mix / mix.sum()
        for i in range(cohort_size):
            learner_id = f"{cohort}-s{i:03d}"
            channel = _CHANNELS[rng.choice(len(_CHANNELS), p=mix)]
            diligence = float(rng.normal(0.0, 1.0))
            started = start + int(rng.integers(0, 2 * DAY))
            enrollments.append({
                "learner_id": learner_id,
                "cohort": cohort,
                "started_at": started,
                "ended_at": None if ongoing else started + duration,
                "channel": channel,
            })
            sessions = max(0, int(rng.poisson(max(0.5, 6 + 3 * diligence))))
            first_14d_minutes = 0
            for _ in range(sessions):
                occurred = started + int(rng.integers(0, 30 * DAY))
                minutes = int(max(5, rng.normal(35, 10)))
                if occurred <= started + 14 * DAY:
                    first_14d_minutes += minutes
                # 约 15% 的活动延迟 5–12 天才入账，模拟补录
                lag = int(rng.integers(0, DAY)) if rng.random() < 0.85 else int(rng.integers(5 * DAY, 12 * DAY))
                activity.append({"learner_id": learner_id, "occurred_at": occurred, "recorded_at": occurred + lag, "minutes": minutes})
            if ongoing:
                continue
            logit = -1.5 + 0.012 * first_14d_minutes + _CHANNEL_EFFECT[channel] + 0.1 * (c - 2.5)
            completed = rng.random() < 1.0 / (1.0 + np.exp(-logit))
            if rng.random() < 0.08:
                outcomes.append({"learner_id": learner_id, "status": "withdrawn", "recorded_at": started + duration + 2 * DAY})
                continue
            outcomes.append({
                "learner_id": learner_id,
                "status": "completed" if completed else "failed",
                "recorded_at": started + duration + 2 * DAY,
            })
            if rng.random() < 0.03:
                # 结课后修正：约 3% 的结果在 20 天后被更正
                outcomes.append({
                    "learner_id": learner_id,
                    "status": "failed" if completed else "completed",
                    "recorded_at": started + duration + 22 * DAY,
                    "corrects": True,
                })
    now = t0 + (n_cohorts - 1) * step + 40 * DAY
    return {
        "enrollments": enrollments,
        "activity": activity,
        "outcomes": outcomes,
        "seats": {f"batch-{c + 1}": 5 for c in range(n_cohorts)},
        "now": now,
        "lead_days": lead_days,
    }
