"""模型登记：候选版本与正式指针分开存放。

训练只写入 candidates/ 下的新版本（内容寻址，重复训练同一数据得到同一
版本号）；正式流程只读 champion.json 指向的版本。promote 是显式的人工
动作，重新训练不会改动正式指针，也不会影响已确认的辅导名单。
"""

import hashlib
import json
import subprocess
from pathlib import Path

import joblib


def fingerprint(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


class ModelRegistry:
    def __init__(self, root) -> None:
        self.root = Path(root)
        (self.root / "candidates").mkdir(parents=True, exist_ok=True)

    def save_candidate(self, model, metadata: dict) -> str:
        payload = {"model": model.name, "feature_names": list(model.feature_names), **metadata}
        version = "v" + fingerprint(payload)[:12]
        target = self.root / "candidates" / version
        if not target.exists():
            target.mkdir()
            model.version = version
            joblib.dump(model, target / "model.joblib")
            (target / "metadata.json").write_text(
                json.dumps({"version": version, **payload}, ensure_ascii=False, indent=2, sort_keys=True)
            )
        return version

    def load(self, version: str):
        target = self.root / "candidates" / version / "model.joblib"
        if not target.exists():
            raise FileNotFoundError(f"候选模型不存在: {version}")
        model = joblib.load(target)
        model.version = version
        return model

    def candidates(self) -> list[str]:
        return sorted(path.name for path in (self.root / "candidates").iterdir() if path.is_dir())

    def promote(self, version: str) -> str:
        if version not in self.candidates():
            raise ValueError(f"不能晋升不存在的候选: {version}")
        (self.root / "champion.json").write_text(json.dumps({"version": version}, ensure_ascii=False))
        return version

    def champion_version(self) -> str | None:
        pointer = self.root / "champion.json"
        return json.loads(pointer.read_text())["version"] if pointer.exists() else None

    def champion(self):
        version = self.champion_version()
        if version is None:
            raise RuntimeError("尚未指定正式模型：请先评估候选，再执行 promote")
        return self.load(version)
