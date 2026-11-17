"""按课程批次回放历史，检验区分、校准与席位选择表现。

回放规则：第 k 个批次在其预测时点训练模型（只用更早批次、且结果在训练
时点前入账的记录），再预测第 k 批；同一批次的记录绝不随机拆开。报告同时
给出区分度、校准、既定席位下的名单与各群体漏选情况，并标注样本支持是否
充足。报告只描述预测表现：预测风险不等于辅导效果，不包含任何因果结论。
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .advising import AdvisingDesk
from .dataset import build_examples
from .models import MODEL_NAMES, TrainedModel

DAY = 86400
DISCLAIMER = "预测风险不等于辅导效果；本报告只描述预测表现，不包含因果结论。"


@dataclass
class ReplayConfig:
    lead_days: int = 14                # 开课后多少天生成预测
    seats: object = 5                  # 每班席位：整数或 {cohort: n}
    group_attributes: tuple = ()       # 额外切片属性；空表示自动发现
    min_fold_n: int = 30               # 单批评估样本下限
    min_fold_failures: int = 10        # 单批失败样本下限
    min_group_failures: int = 5        # 群体下结论所需失败样本下限
    disparity_margin: float = 0.10     # 捕获率低于整体多少算“漏选明显”
    bootstrap: int = 1000
    seed: int = 0

    def seat_count(self, cohort: str) -> int:
        return int(self.seats.get(cohort, 0)) if isinstance(self.seats, dict) else int(self.seats)


def _batch_plan(enrollments: list[dict], lead_days: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in enrollments:
        groups.setdefault(row["cohort"], []).append(row)
    plan = []
    for cohort, rows in groups.items():
        start = min(int(row["started_at"]) for row in rows)
        plan.append({
            "cohort": cohort,
            "rows": sorted(rows, key=lambda row: row["learner_id"]),
            "start": start,
            "predict_as_of": start + lead_days * DAY,
        })
    plan.sort(key=lambda batch: (batch["start"], batch["cohort"]))
    return plan


def _metric_row(y: list[int], p: list[float]) -> dict:
    n = len(y)
    failures = sum(1 for value in y if value == 0)
    row = {"n": n, "failures": failures, "auc": None, "average_precision": None, "brier": None, "log_loss": None}
    if n == 0:
        return row
    row["brier"] = float(brier_score_loss(y, p))
    if 0 < failures < n:
        row["auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        row["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return row


def _bootstrap(y: list[int], p: list[float], config: ReplayConfig) -> dict:
    if not y:
        return {"auc_ci95": None, "brier_ci95": None}
    rng = np.random.default_rng(config.seed)
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    n = len(y_arr)
    aucs, briers = [], []
    for _ in range(config.bootstrap):
        idx = rng.integers(0, n, n)
        sample_y = y_arr[idx]
        briers.append(brier_score_loss(sample_y, p_arr[idx]))
        if 0 < int(sample_y.sum()) < n:
            aucs.append(roc_auc_score(sample_y, p_arr[idx]))

    def interval(values):
        if not values:
            return None
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"auc_ci95": interval(aucs), "brier_ci95": interval(briers)}


def _calibration(y: list[int], p: list[float], bins: int = 10) -> list[dict]:
    table = []
    for bucket in range(bins):
        lo, hi = bucket / bins, (bucket + 1) / bins
        members = [i for i, value in enumerate(p) if lo <= value < hi or (bucket == bins - 1 and value == 1.0)]
        if members:
            table.append({
                "bin": bucket,
                "lo": lo,
                "hi": hi,
                "n": len(members),
                "mean_predicted": float(np.mean([p[i] for i in members])),
                "observed": float(np.mean([y[i] for i in members])),
            })
    return table


def _group_keys(example: dict, attr_keys: list[str]) -> list[str]:
    keys = [f"cohort={example['cohort']}"]
    for attr in attr_keys:
        if attr in example["attributes"]:
            keys.append(f"{attr}={example['attributes'][attr]}")
    return keys


def _selection_report(groups: dict, overall: dict, selection: list[dict], config: ReplayConfig) -> dict:
    overall_capture = overall["captured"] / overall["failures"] if overall["failures"] else None
    rows = {}
    flags = []
    for key, stat in sorted(groups.items()):
        capture = stat["captured"] / stat["failures"] if stat["failures"] else None
        row = {**stat, "missed": stat["failures"] - stat["captured"], "capture_rate": capture}
        if stat["failures"] < config.min_group_failures:
            row["support"] = "样本不足"
        else:
            row["support"] = "ok"
            if overall_capture is not None and capture is not None and overall_capture - capture > config.disparity_margin:
                flags.append(
                    f"群体 {key} 漏选明显：失败学员捕获率 {capture:.2f} 低于整体 {overall_capture:.2f}"
                    f"（失败样本 {stat['failures']}，漏选 {row['missed']} 人）"
                )
        rows[key] = row
    return {"overall_capture_rate": overall_capture, "groups": rows, "flags": flags, "folds": selection}


def _replay_model(name: str, folds: list[dict], activity: list[dict], config: ReplayConfig) -> dict:
    fold_rows = []
    pooled_y: list[int] = []
    pooled_p: list[float] = []
    groups: dict[str, dict] = {}
    overall = {"failures": 0, "captured": 0}
    selection = []
    attr_keys = list(config.group_attributes)
    if not attr_keys:
        found = set()
        for fold in folds:
            for example in fold["test"]:
                found |= set(example["attributes"])
        attr_keys = sorted(found)
    for fold in folds:
        try:
            model = TrainedModel.fit(name, fold["train"])
        except ValueError as exc:
            fold_rows.append({"cohort": fold["cohort"], "train_cohorts": fold["train_cohorts"], "skipped": str(exc)})
            continue
        labeled = [example for example in fold["test"] if example["label"] is not None]
        excluded: dict[str, int] = {}
        for example in fold["test"]:
            if example["excluded"]:
                excluded[example["excluded"]] = excluded.get(example["excluded"], 0) + 1
        label_by_id = {example["learner_id"]: example["label"] for example in fold["test"]}
        desk = AdvisingDesk({fold["cohort"]: config.seat_count(fold["cohort"])})
        proposals = desk.propose(fold["enrollments"], activity, fold["predict_as_of"], model=model)
        selected_ids = {proposal["learner_id"] for proposal in proposals}
        score_by_id = {example["learner_id"]: model.predict_proba_one(example["features"]) for example in fold["test"]}
        y = [example["label"] for example in labeled]
        p = [score_by_id[example["learner_id"]] for example in labeled]
        row = _metric_row(y, p)
        row.update({
            "cohort": fold["cohort"],
            "train_cohorts": fold["train_cohorts"],
            "n_enrolled": len(fold["test"]),
            "n_labeled": len(labeled),
            "excluded": excluded,
            "selected": len(proposals),
            "support": "ok" if len(labeled) >= config.min_fold_n and row["failures"] >= config.min_fold_failures else "样本不足",
        })
        fold_rows.append(row)
        pooled_y.extend(y)
        pooled_p.extend(p)
        for example in labeled:
            is_failure = example["label"] == 0
            is_selected = example["learner_id"] in selected_ids
            if is_failure:
                overall["failures"] += 1
                if is_selected:
                    overall["captured"] += 1
            for key in _group_keys(example, attr_keys):
                stat = groups.setdefault(key, {"n": 0, "failures": 0, "selected": 0, "captured": 0})
                stat["n"] += 1
                stat["failures"] += 1 if is_failure else 0
                stat["selected"] += 1 if is_selected else 0
                stat["captured"] += 1 if (is_selected and is_failure) else 0
        selection.append({
            "cohort": fold["cohort"],
            "seats": config.seat_count(fold["cohort"]),
            "selected": [
                {"learner_id": proposal["learner_id"], "score": round(proposal["score"], 6), "label": label_by_id.get(proposal["learner_id"])}
                for proposal in proposals
            ],
        })
    pooled = _metric_row(pooled_y, pooled_p)
    pooled.update(_bootstrap(pooled_y, pooled_p, config))
    return {
        "folds": fold_rows,
        "pooled": pooled,
        "calibration": _calibration(pooled_y, pooled_p),
        "selection": _selection_report(groups, overall, selection, config),
    }


def _compare(models: dict, order: list) -> dict:
    if len(order) < 2:
        return {"notes": ["只有一个模型，没有候选比较"]}
    reference, candidate = order[0], order[1]
    ref, cand = models[reference], models[candidate]
    deltas = {}
    notes = []
    if ref["pooled"].get("auc") is not None and cand["pooled"].get("auc") is not None:
        deltas["auc"] = cand["pooled"]["auc"] - ref["pooled"]["auc"]
        notes.append(f"候选 {candidate} 相对基线 {reference} 的 AUC 差 {deltas['auc']:+.3f}；请结合逐批稳定性与置信区间判断，不要只看平均值")
    if ref["pooled"].get("brier") is not None and cand["pooled"].get("brier") is not None:
        deltas["brier"] = cand["pooled"]["brier"] - ref["pooled"]["brier"]
        notes.append(f"Brier 差 {deltas['brier']:+.3f}（负值表示候选校准更好）")
    new_flags = sorted(set(cand["selection"]["flags"]) - set(ref["selection"]["flags"]))
    for flag in new_flags:
        notes.append(f"候选新增漏选信号：{flag}")
    notes.append("是否晋升由人工决定；本工具不自动替换正式模型。预测风险不等于辅导效果。")
    return {"reference": reference, "candidate": candidate, "deltas": deltas, "notes": notes}


def _warnings(models: dict, config: ReplayConfig) -> list[str]:
    warnings = []
    for name, report in models.items():
        failures = report["pooled"].get("failures", 0)
        if failures < 2 * config.min_fold_failures:
            warnings.append(f"{name}: 失败样本总量 {failures} 偏少，置信区间与比较结论仅供参考")
        low = [fold["cohort"] for fold in report["folds"] if fold.get("support") == "样本不足"]
        if low:
            warnings.append(f"{name}: 批次 {', '.join(low)} 样本不足，逐批指标仅供参考")
    return warnings


def replay(enrollments: list[dict], activity: list[dict], outcomes: list[dict],
           config: ReplayConfig | None = None, model_names=MODEL_NAMES,
           eval_label_as_of: int | None = None) -> dict:
    config = config or ReplayConfig()
    plan = _batch_plan(enrollments, config.lead_days)
    if eval_label_as_of is None:
        eval_label_as_of = max((int(record["recorded_at"]) for record in outcomes), default=0)
    folds = []
    for k in range(1, len(plan)):
        test = plan[k]
        train_examples = []
        for earlier in plan[:k]:
            train_examples.extend(build_examples(earlier["rows"], activity, outcomes, earlier["predict_as_of"], test["predict_as_of"]))
        folds.append({
            "cohort": test["cohort"],
            "predict_as_of": test["predict_as_of"],
            "train_cohorts": [batch["cohort"] for batch in plan[:k]],
            "train": train_examples,
            "test": build_examples(test["rows"], activity, outcomes, test["predict_as_of"], eval_label_as_of),
            "enrollments": test["rows"],
        })
    models = {name: _replay_model(name, folds, activity, config) for name in model_names}
    return {
        "eval_label_as_of": eval_label_as_of,
        "config": {
            "lead_days": config.lead_days,
            "seats": config.seats,
            "min_fold_n": config.min_fold_n,
            "min_fold_failures": config.min_fold_failures,
            "min_group_failures": config.min_group_failures,
            "disparity_margin": config.disparity_margin,
            "bootstrap": config.bootstrap,
        },
        "batches": [
            {"cohort": batch["cohort"], "start": batch["start"], "predict_as_of": batch["predict_as_of"], "n": len(batch["rows"])}
            for batch in plan
        ],
        "models": models,
        "comparison": _compare(models, list(model_names)),
        "warnings": _warnings(models, config),
        "disclaimer": DISCLAIMER,
    }


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_ci(interval):
    return "—" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def format_summary(report: dict) -> str:
    lines = ["批次回放（训练只用更早批次，检验按时间先后推进）"]
    for batch in report["batches"]:
        lines.append(f"  {batch['cohort']}: start={batch['start']} predict_as_of={batch['predict_as_of']} n={batch['n']}")
    for name, model in report["models"].items():
        pooled = model["pooled"]
        lines.append(f"\n模型 {name}")
        lines.append(
            f"  汇总: n={pooled['n']} 失败={pooled['failures']}"
            f" AUC={_fmt(pooled['auc'])} {_fmt_ci(pooled.get('auc_ci95'))}"
            f" AP={_fmt(pooled['average_precision'])}"
            f" Brier={_fmt(pooled['brier'])} {_fmt_ci(pooled.get('brier_ci95'))}"
            f" log_loss={_fmt(pooled['log_loss'])}"
        )
        for fold in model["folds"]:
            if "skipped" in fold:
                lines.append(f"  {fold['cohort']}: 跳过（{fold['skipped']}）")
                continue
            lines.append(
                f"  {fold['cohort']}: 标记={fold['n_labeled']}/{fold['n_enrolled']} 失败={fold['failures']}"
                f" AUC={_fmt(fold['auc'])} Brier={_fmt(fold['brier'])} 支持={fold['support']}"
                + (f" 排除={fold['excluded']}" if fold["excluded"] else "")
            )
        calibration = " ".join(
            f"[{row['lo']:.1f},{row['hi']:.1f})p={row['mean_predicted']:.2f}/o={row['observed']:.2f}/n={row['n']}"
            for row in model["calibration"]
        )
        lines.append(f"  校准(预测/实际): {calibration or '无'}")
        selection = model["selection"]
        lines.append(f"  席位选择: 整体失败学员捕获率={_fmt(selection['overall_capture_rate'], 2)}")
        for fold in selection["folds"]:
            chosen = ", ".join(f"{row['learner_id']}({row['score']:.2f})" for row in fold["selected"]) or "无"
            lines.append(f"    {fold['cohort']} 席位={fold['seats']}: {chosen}")
        lines.append(f"  漏选信号: {'；'.join(selection['flags']) if selection['flags'] else '无'}")
    lines.append("\n候选比较")
    for note in report["comparison"]["notes"]:
        lines.append(f"  {note}")
    if report["warnings"]:
        lines.append("\n警告")
        for warning in report["warnings"]:
            lines.append(f"  {warning}")
    lines.append(f"\n{report['disclaimer']}")
    return "\n".join(lines)
