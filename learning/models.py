"""可训练模型：可解释基线（逻辑回归）与候选方法（梯度提升树）。

两个模型共用同一套时点特征口径（learning.features.FEATURE_NAMES），
训练与席位流程使用同一个 TrainedModel 对象，避免线上线下不一致。
分数为预测完成概率，越低表示风险越高、越优先辅导。
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_NAMES

BASELINE = "logistic_baseline"
CANDIDATE = "gbdt_candidate"
MODEL_NAMES = (BASELINE, CANDIDATE)


def _new_estimator(name: str):
    if name == BASELINE:
        return LogisticRegression(max_iter=1000, random_state=0)
    if name == CANDIDATE:
        return HistGradientBoostingClassifier(random_state=0, early_stopping=False)
    raise ValueError(f"未知模型: {name}")


class TrainedModel:
    """已拟合模型及其特征口径，供评估与席位流程直接使用。"""

    def __init__(self, name: str, estimator, feature_names: tuple = FEATURE_NAMES, version: str | None = None) -> None:
        self.name = name
        self.estimator = estimator
        self.feature_names = tuple(feature_names)
        self.version = version

    @classmethod
    def fit(cls, name: str, examples: list[dict]) -> "TrainedModel":
        labeled = [row for row in examples if row["label"] is not None]
        if {row["label"] for row in labeled} != {0, 1}:
            raise ValueError("训练样本需要同时包含完成与未完成两类")
        estimator = _new_estimator(name)
        estimator.fit(
            [[row["features"][feature] for feature in FEATURE_NAMES] for row in labeled],
            [row["label"] for row in labeled],
        )
        return cls(name, estimator)

    def predict_proba_one(self, features: dict) -> float:
        row = [[features[feature] for feature in self.feature_names]]
        return float(self.estimator.predict_proba(row)[0][1])

    def predict_proba(self, feature_rows: list[dict]) -> list[float]:
        return [self.predict_proba_one(row) for row in feature_rows]

    def coefficients(self) -> dict[str, float] | None:
        """基线模型的可解释系数；候选模型没有此口径，返回 None。"""
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            return None
        weights = {feature: float(value) for feature, value in zip(self.feature_names, coef[0])}
        weights["intercept"] = float(self.estimator.intercept_[0])
        return weights
