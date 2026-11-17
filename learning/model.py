def score(features: dict[str, float | int], parameters: dict[str, float]) -> float:
    value = parameters.get("intercept", 0.0)
    for name, feature in features.items():
        value += parameters.get(name, 0.0) * float(feature)
    return 1.0 / (1.0 + pow(2.718281828, -max(-30.0, min(30.0, value))))
