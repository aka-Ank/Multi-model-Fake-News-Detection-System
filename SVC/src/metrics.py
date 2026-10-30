"""Binary and multiclass classification metrics using standard Python only."""


def confusion_matrix(y_true, y_pred, labels=None):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: position for position, label in enumerate(labels)}
    for truth, prediction in zip(y_true, y_pred):
        if truth not in index or prediction not in index:
            raise ValueError("all values must occur in labels")
        matrix[index[truth]][index[prediction]] += 1
    return labels, matrix


def classification_report(y_true, y_pred, labels=None, zero_division=0.0):
    labels, matrix = confusion_matrix(y_true, y_pred, labels)
    total, correct, per_class = len(y_true), sum(matrix[i][i] for i in range(len(labels))), {}
    recalls, f1s, supports = [], [], []
    for index, label in enumerate(labels):
        tp = matrix[index][index]; fp = sum(matrix[row][index] for row in range(len(labels))) - tp
        fn = sum(matrix[index]) - tp; support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else zero_division
        recall = tp / (tp + fn) if tp + fn else zero_division
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else zero_division
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall); f1s.append(f1)
        supports.append(support)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1s, supports)) / total if total else 0.0
    return {"labels": labels, "confusion_matrix": matrix, "accuracy": correct / total if total else 0.0,
            "per_class": per_class, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0}
