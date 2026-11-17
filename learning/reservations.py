"""已确认名单的本地存证：同号不同名单会被拒绝。

与 AdvisingDesk.reserve 同一口径，但落盘保存，跨进程、跨训练轮次有效；
重新训练产生的新名单不能覆盖已确认的场次。
"""

import json
from pathlib import Path


class ReservationStore:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(self, session_id: str, proposals: list[dict]) -> list[dict]:
        path = self.root / f"{session_id}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != proposals:
                raise ValueError("已确认的辅导名单不能被同号覆盖")
            return existing
        path.write_text(json.dumps(proposals, ensure_ascii=False, indent=2, sort_keys=True))
        return proposals

    def get(self, session_id: str) -> list[dict] | None:
        path = self.root / f"{session_id}.json"
        return json.loads(path.read_text()) if path.exists() else None
