# === file: src/models/rf.py ===
from sklearn.ensemble import RandomForestClassifier
import joblib

__all__ = ["RandomForestFallDetector"]

class RandomForestFallDetector:
    """
    Wrapper for a RandomForestClassifier for fall detection.
    - Uses balanced class weights and strong regularization.
    - Provides fit, predict, predict_proba, save, and load methods.
    """
    def __init__(self):
        # Initialize the RandomForestClassifier with specified hyperparameters
        self.clf = RandomForestClassifier(
            n_estimators=1200,      # Number of trees in the forest
            max_depth=20,           # Maximum depth of each tree
            min_samples_leaf=5,     # Minimum samples required at a leaf node
            n_jobs=-1,              # Use all available CPU cores
            class_weight="balanced",# Handle class imbalance
            random_state=42,        # For reproducibility
        )

    def fit(self, X, y):
        """
        Fit the RandomForest model to the training data.
        """
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        return self.clf.predict_proba(X)

    # persistence
    def save(self, path):
        """
        Save the trained model to disk using joblib.
        """
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        """
        Load a trained model from disk.
        """
        obj = cls()
        obj.clf = joblib.load(path)
        return obj
