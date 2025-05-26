from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(y_true, y_pred, average: str = "macro"):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}