# === file: src/models/rf.py ===
from sklearn.ensemble import RandomForestClassifier
import joblib

__all__ = ["RandomForestFallDetector"]

class RandomForestFallDetector:
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=1200,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):       return self.clf.predict(X)
    def predict_proba(self, X): return self.clf.predict_proba(X)

    # persistence
    def save(self, path): joblib.dump(self.clf, path)
    @classmethod
    def load(cls, path):
        obj = cls(); obj.clf = joblib.load(path); return obj
