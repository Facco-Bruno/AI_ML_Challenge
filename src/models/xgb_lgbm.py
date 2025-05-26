"""
Stable XGBoost + LightGBM trainers (no class-weights, no custom tricks).
"""

from __future__ import annotations
import lightgbm as lgb
from xgboost import XGBClassifier

__all__ = ["train_xgboost", "train_lightgbm"]


def train_xgboost(X_tr, y_tr, X_val, y_val):
    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    model.set_params(n_estimators=model.best_ntree_limit)
    return model


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    train = lgb.Dataset(X_tr, y_tr)
    valid = lgb.Dataset(X_val, y_val, reference=train)
    params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42,
    )
    return lgb.train(
        params,
        train,
        num_boost_round=1000,
        valid_sets=[valid],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
