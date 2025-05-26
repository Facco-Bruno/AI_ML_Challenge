# === file: src/models/xgb_lgbm.py ===
from xgboost import XGBClassifier
import lightgbm as lgb

__all__ = ["train_xgboost", "train_lightgbm"]

def train_xgboost(X_tr, y_tr, X_val, y_val):
    """
    Trains an XGBoost classifier for multiclass classification.
    - Uses early stopping on the validation set.
    - Returns the trained model with best iteration.
    """
    model = XGBClassifier(
        n_estimators=1000,         # Maximum number of boosting rounds
        max_depth=8,               # Maximum tree depth
        learning_rate=0.05,        # Step size shrinkage
        subsample=0.9,             # Row sampling
        colsample_bytree=0.8,      # Feature sampling
        min_child_weight=5,        # Minimum sum of instance weight in a child
        gamma=0.5,                 # Minimum loss reduction for split
        reg_lambda=1.0,            # L2 regularization
        objective="multi:softprob",# Multiclass probability output
        num_class=3,               # Number of classes
        n_jobs=-1,                 # Use all CPU cores
        random_state=42,           # For reproducibility
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100, # Stop if no improvement after 100 rounds
        verbose=False,
    )
    model.set_params(n_estimators=model.best_ntree_limit) # Use best iteration
    return model

def train_lightgbm(X_tr, y_tr, X_val, y_val):
    """
    Trains a LightGBM model for multiclass classification.
    - Uses early stopping on the validation set.
    - Returns the trained booster.
    """
    train = lgb.Dataset(X_tr, y_tr)
    valid = lgb.Dataset(X_val, y_val, reference=train)
    params = dict(
        objective="multiclass",    # Multiclass objective
        num_class=3,               # Number of classes
        metric="multi_logloss",    # Evaluation metric
        learning_rate=0.05,        # Step size shrinkage
        num_leaves=31,             # Number of leaves in one tree
        subsample=0.8,             # Row sampling
        colsample_bytree=0.8,      # Feature sampling
        seed=42,                   # For reproducibility
    )
    return lgb.train(
        params, train, num_boost_round=1000,
        valid_sets=[valid],
        early_stopping_rounds=100, # Stop if no improvement after 100 rounds
        verbose_eval=False,
    )
