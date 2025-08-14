from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
@dataclass
class AnomalyDetector:
    model: IsolationForest
    @classmethod
    def train_from_baseline(cls, df: pd.DataFrame, contamination: float = 0.02) -> "AnomalyDetector":
        data = pd.get_dummies(df, columns=["metric"], drop_first=False)
        X = data[[c for c in data.columns if c not in {"timestamp","sensor_id","unit"}]].fillna(0.0).to_numpy()
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42); model.fit(X); return cls(model)
    def score(self, row: pd.DataFrame) -> float:
        data = pd.get_dummies(row, columns=["metric"], drop_first=False)
        X = data[[c for c in data.columns if c not in {"timestamp","sensor_id","unit"}]].to_numpy()
        s = -self.model.score_samples(X)[0]; return float(s)
    def predict_flag(self, row: pd.DataFrame, threshold: float = 0.6) -> bool:
        return self.score(row) > threshold
