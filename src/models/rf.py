"""
Random-Forest wrapper (simple, no sample_weight logic inside).
"""

from sklearn.ensemble import RandomForestClassifier
import joblib

__all__ = ["RandomForestFallDetector"]


class RandomForestFallDetector:
    def __init__(self, **kw):
        base = dict(
            n_estimators=600,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        base.update(kw)
        self.clf = RandomForestClassifier(**base)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def save(self, path):
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.clf = joblib.load(path)
        return obj
