"""Logistic catch heads with independent held-out Platt calibration."""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .selection import CRITICS

LEXICONS = {
    "ledger": {"gave", "received", "remaining", "bought", "sold", "each", "total"},
    "ratio_basis": {"ratio", "percent", "percentage", "discount", "per", "fraction"},
    "constraint": {"least", "most", "integer", "positive", "exactly", "must"},
    "arithmetic": {"sum", "product", "difference", "divide", "multiply", "calculate"},
    "countermodel": {"assume", "possible", "otherwise", "if", "alternative"},
}


def features(text, history, consumed_ratio):
    if not 0 <= consumed_ratio <= 1:
        raise ValueError("Consumed-budget ratio must lie in [0, 1]")
    words = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    result = {f"word:{word}": float(count) for word, count in words.items()}
    result["budget_ratio"] = float(consumed_ratio)
    for niche, lexicon in LEXICONS.items():
        result[f"lexical:{niche}"] = float(sum(words[word] for word in lexicon))
    for index, event in enumerate(history):
        critic, action = event["critic"], event["action"]
        if action not in {"KEEP", "REPLACE", "INVALID"}:
            raise ValueError("Invalid history action")
        result[f"prior:{critic}:{action}"] = 1.0
        result[f"position:{index}:{critic}:{action}"] = 1.0
        if event.get("verified") is not None:
            result[f"verified:{critic}:{bool(event['verified'])}"] = 1.0
    return result


@dataclass(frozen=True)
class RouterExample:
    problem_id: str
    text: str
    history: list[dict]
    consumed_ratio: float
    catches: dict[str, bool]


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


class CatchRouter:
    def __init__(self, vocabulary, heads, fit_ids=(), calibration_ids=()):
        self.vocabulary = tuple(vocabulary)
        self.heads = heads
        self.fit_ids = tuple(fit_ids)
        self.calibration_ids = tuple(calibration_ids)

    @classmethod
    def fit(cls, training, calibration, critics=CRITICS, seed=42):
        training, calibration, critics = list(training), list(calibration), tuple(critics)
        if not training or not calibration:
            raise ValueError("Both fitting and calibration examples are required")
        fit_ids = {r.problem_id for r in training}
        cal_ids = {r.problem_id for r in calibration}
        if fit_ids & cal_ids:
            raise ValueError("Fitting and calibration problem IDs must be disjoint")
        if not critics or len(set(critics)) != len(critics):
            raise ValueError("Expected distinct critic names")
        for row in training + calibration:
            if any(type(row.catches.get(c)) is not bool for c in critics):
                raise ValueError("Every example needs a boolean native-catch target per critic")
        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform(features(r.text, r.history, r.consumed_ratio) for r in training)
        xc = vectorizer.transform(features(r.text, r.history, r.consumed_ratio) for r in calibration)
        heads = {}
        for critic in critics:
            y = np.array([r.catches[critic] for r in training], dtype=int)
            yc = np.array([r.catches[critic] for r in calibration], dtype=int)
            if len(np.unique(y)) < 2 or len(np.unique(yc)) < 2:
                raise ValueError(f"Critic {critic!r} needs both catch classes in fitting and calibration data")
            native = LogisticRegression(random_state=seed, max_iter=2000).fit(x, y)
            scores = native.decision_function(xc).reshape(-1, 1)
            platt = LogisticRegression(C=1e6, random_state=seed, max_iter=2000).fit(scores, yc)
            heads[critic] = {
                "weights": native.coef_[0].tolist(), "intercept": float(native.intercept_[0]),
                "platt_slope": float(platt.coef_[0, 0]), "platt_intercept": float(platt.intercept_[0]),
            }
        return cls(vectorizer.get_feature_names_out().tolist(), heads, sorted(fit_ids), sorted(cal_ids))

    def scores(self, text, history, consumed_ratio, candidates):
        f = features(text, history, consumed_ratio)
        result = {}
        for critic in candidates:
            head = self.heads[critic]
            raw = head["intercept"] + sum(w * f.get(name, 0.0) for name, w in zip(self.vocabulary, head["weights"]))
            result[critic] = sigmoid(head["platt_slope"] * raw + head["platt_intercept"])
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({"version": 1, "vocabulary": self.vocabulary,
            "heads": self.heads, "fit_ids": self.fit_ids, "calibration_ids": self.calibration_ids}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported router format")
        vocabulary = data["vocabulary"]
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Duplicate router feature names")
        for head in data["heads"].values():
            values = head["weights"] + [head["intercept"], head["platt_slope"], head["platt_intercept"]]
            if len(head["weights"]) != len(vocabulary) or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid router parameters")
        return cls(vocabulary, data["heads"], data.get("fit_ids", ()), data.get("calibration_ids", ()))
