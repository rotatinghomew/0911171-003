"""可重现的本地路径：evaluate → train → promote → propose。

- evaluate：按课程批次回放历史，比较可解释基线与候选方法，只读不写；
- train：把基线与候选写入 candidates/（内容寻址，同数据同版本号）；
- promote：显式把某个候选设为正式模型（人工动作，工具不自动晋升）；
- propose：用正式模型生成班次名单，可写入已确认名单存证。

候选比较与正式预留分开；重新训练只产生新候选版本，不会改动正式指针
或已确认的辅导名单。预测风险不等于辅导效果。
"""

import argparse
import json
import sys
from pathlib import Path

from . import sample_data
from .advising import AdvisingDesk
from .dataset import build_examples
from .evaluation import DAY, ReplayConfig, _batch_plan, format_summary, replay
from .models import MODEL_NAMES, TrainedModel
from .registry import ModelRegistry, _git_revision, fingerprint
from .reservations import ReservationStore


def load_data(source: str) -> dict:
    """--data synthetic 使用内置合成历史；否则读取目录下的 JSON 记录。"""
    if source == "synthetic":
        return sample_data.generate()
    root = Path(source)
    data = {name: json.loads((root / f"{name}.json").read_text()) for name in ("enrollments", "activity", "outcomes")}
    seats = root / "seats.json"
    data["seats"] = json.loads(seats.read_text()) if seats.exists() else {}
    meta = root / "meta.json"
    data["now"] = json.loads(meta.read_text()).get("now") if meta.exists() else None
    data.setdefault("lead_days", 14)
    return data


def run_evaluate(data: dict, config: ReplayConfig) -> dict:
    return replay(
        data["enrollments"],
        data["activity"],
        data["outcomes"],
        config=config,
        eval_label_as_of=data.get("now") or None,
    )


def run_train(data: dict, artifacts_dir, label_as_of: int | None = None, lead_days: int | None = None) -> dict:
    lead_days = lead_days or data.get("lead_days", 14)
    if label_as_of is None:
        label_as_of = data.get("now") or max((int(r["recorded_at"]) for r in data["outcomes"]), default=0)

    examples = []
    for batch in _batch_plan(data["enrollments"], lead_days):
        examples.extend(build_examples(batch["rows"], data["activity"], data["outcomes"], batch["predict_as_of"], label_as_of))
    labeled = [row for row in examples if row["label"] is not None]
    registry = ModelRegistry(artifacts_dir)
    data_hash = fingerprint({"enrollments": data["enrollments"], "activity": data["activity"], "outcomes": data["outcomes"]})
    versions = {}
    for name in MODEL_NAMES:
        model = TrainedModel.fit(name, examples)
        metadata = {
            "label_as_of": label_as_of,
            "lead_days": lead_days,
            "train_examples": len(labeled),
            "train_failures": sum(1 for row in labeled if row["label"] == 0),
            "data_sha256": data_hash,
            "git": _git_revision(),
        }
        versions[name] = registry.save_candidate(model, metadata)
    return {
        "versions": versions,
        "label_as_of": label_as_of,
        "lead_days": lead_days,
        "train_examples": len(labeled),
        "train_failures": sum(1 for row in labeled if row["label"] == 0),
    }


def run_propose(data: dict, artifacts_dir, cohort: str, as_of: int | None = None, lead_days: int | None = None,
                session: str | None = None, reservations_dir=None) -> dict:
    registry = ModelRegistry(artifacts_dir)
    model = registry.champion()
    lead_days = lead_days or data.get("lead_days", 14)
    rows = [row for row in data["enrollments"] if row["cohort"] == cohort]
    if not rows:
        raise ValueError(f"班次不存在: {cohort}")
    if as_of is None:
        as_of = min(int(row["started_at"]) for row in rows) + lead_days * DAY
    seats = int((data.get("seats") or {}).get(cohort, 0))
    desk = AdvisingDesk({cohort: seats})
    proposals = desk.propose(rows, data["activity"], as_of, model=model)
    payload = {"cohort": cohort, "as_of": as_of, "seats": seats, "model_version": model.version, "proposals": proposals}
    if session and reservations_dir:
        payload["reserved"] = ReservationStore(reservations_dir).reserve(session, proposals)
    return payload


def _parse_seats(raw, data):
    if raw is None:
        return data.get("seats") or 5
    if raw.isdigit():
        return int(raw)
    return json.loads(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="learning.pipeline", description="训练到名单的可重现路径")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("evaluate", help="回放历史批次，比较基线与候选")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--out", default=None, help="报告 JSON 输出路径")
    p.add_argument("--lead-days", type=int, default=None)
    p.add_argument("--seats", default=None, help="整数或 JSON 字典，默认取数据内 seats")
    p.add_argument("--bootstrap", type=int, default=1000)

    p = commands.add_parser("train", help="训练基线与候选，写入候选版本")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--label-as-of", type=int, default=None)
    p.add_argument("--lead-days", type=int, default=None)

    p = commands.add_parser("promote", help="把某个候选版本设为正式模型")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--version", required=True)

    p = commands.add_parser("propose", help="用正式模型生成班次名单")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--cohort", required=True)
    p.add_argument("--as-of", type=int, default=None)
    p.add_argument("--lead-days", type=int, default=None)
    p.add_argument("--session", default=None, help="确认名单的场次编号")
    p.add_argument("--reservations", default=None, help="已确认名单的存证目录")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        data = load_data(args.data)
        config = ReplayConfig(
            lead_days=args.lead_days or data.get("lead_days", 14),
            seats=_parse_seats(args.seats, data),
            bootstrap=args.bootstrap,
        )
        report = run_evaluate(data, config)
        print(format_summary(report))
        if args.out:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train":
        data = load_data(args.data)
        result = run_train(data, args.artifacts, label_as_of=args.label_as_of, lead_days=args.lead_days)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        registry = ModelRegistry(args.artifacts)
        baseline = registry.load(result["versions"]["logistic_baseline"])
        print("基线系数:", json.dumps(baseline.coefficients(), ensure_ascii=False))
        print("提示：候选已写入，是否晋升请依据 evaluate 报告人工决定（promote 命令）。")
        return 0

    if args.command == "promote":
        version = ModelRegistry(args.artifacts).promote(args.version)
        print(f"正式模型已设为 {version}")
        return 0

    if args.command == "propose":
        data = load_data(args.data)
        payload = run_propose(data, args.artifacts, args.cohort, as_of=args.as_of, lead_days=args.lead_days,
                              session=args.session, reservations_dir=args.reservations)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("提醒：预测风险不等于辅导效果，名单不构成因果承诺。", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
