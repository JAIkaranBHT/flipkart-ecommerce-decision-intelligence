"""From-scratch classification routines reused by Parts 2 and 4.

Importing this module only defines functions. It never trains a model or writes
an output file.
"""

import numpy as np


def split_and_standardise(X, y):
    """Use the first 70% of each class, then restore the original row order."""
    train_mask = np.zeros(len(y), dtype=bool)
    for label in (0, 1):
        class_indices = np.flatnonzero(y == label)
        cutoff = 7 * len(class_indices) // 10
        train_mask[class_indices[:cutoff]] = True
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(~train_mask)
    raw_train, raw_test = X[train_indices], X[test_indices]
    train_mean = raw_train.mean(axis=0)
    train_sd = raw_train.std(axis=0, ddof=1)
    if np.any(train_sd == 0):
        raise ValueError("Cannot standardise a constant training feature.")
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "train_mean": train_mean,
        "train_sd": train_sd,
        "X_train": (raw_train - train_mean) / train_sd,
        "X_test": (raw_test - train_mean) / train_sd,
        "y_train": y[train_indices],
        "y_test": y[test_indices],
    }


def sigmoid(scores):
    """Evaluate the sigmoid without overflowing exp for large negative scores."""
    probabilities = np.empty_like(scores, dtype=float)
    nonnegative = scores >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
    exp_scores = np.exp(scores[~nonnegative])
    probabilities[~nonnegative] = exp_scores / (1.0 + exp_scores)
    return probabilities


def predict_probability(X, weights, bias):
    return sigmoid(X @ weights + bias)


def binary_cross_entropy(y, probabilities):
    """Mean binary cross-entropy; clipping only protects the logarithms."""
    p = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_logistic(X, y, learning_rate=0.1, updates=1000):
    """Full-batch gradient descent, with each loss measured after its update."""
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    probabilities = predict_probability(X, weights, bias)
    loss_history = {0: binary_cross_entropy(y, probabilities)}
    for epoch in range(1, updates + 1):
        errors = probabilities - y
        # Both gradients use the same parameters, before either update.
        gradient_weights = X.T @ errors / len(y)
        gradient_bias = float(np.mean(errors))
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        probabilities = predict_probability(X, weights, bias)
        loss_history[epoch] = binary_cross_entropy(y, probabilities)
    return weights, bias, loss_history


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(y_true, predicted):
    actual_positive = np.asarray(y_true) == 1
    predicted_positive = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual_positive & predicted_positive))
    fp = int(np.sum(~actual_positive & predicted_positive))
    tn = int(np.sum(~actual_positive & ~predicted_positive))
    fn = int(np.sum(actual_positive & ~predicted_positive))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
        "specificity": safe_ratio(tn, tn + fp),
        "tpr": recall,
        "fpr": safe_ratio(fp, fp + tn),
    }


def neighbour_labels(reference_X, reference_y, query_X):
    """Rank by Euclidean distance, preserving reference file order for ties."""
    differences = query_X[:, None, :] - reference_X[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    ranked_indices = np.argsort(distances, axis=1, kind="stable")
    return reference_y[ranked_indices]


def knn_probability(reference_X, reference_y, query_X, k):
    labels = neighbour_labels(reference_X, reference_y, query_X)
    return np.mean(labels[:, :k], axis=1)


def select_knn_k(X, y):
    """Mean held-out F1 across the five pinned folds, not pooled-fold F1.

    X is already standardised with all training rows, as the brief requires.
    No scaling is fitted inside these folds.
    """
    candidates = list(range(1, 22, 2))
    fold_scores = {k: [] for k in candidates}
    fold_ids = np.arange(len(y)) % 5
    for fold in range(5):
        held_out = fold_ids == fold
        labels = neighbour_labels(X[~held_out], y[~held_out], X[held_out])
        positive_counts = np.cumsum(labels[:, :max(candidates)], axis=1)
        for k in candidates:
            prediction = positive_counts[:, k - 1] >= (k + 1) // 2
            score = confusion_metrics(y[held_out], prediction)["f1"]
            fold_scores[k].append(score)
    cv_rows = [(k, float(np.mean(fold_scores[k]))) for k in candidates]
    # Rows are in ascending K, so max retains the smaller K for an exact tie.
    best_k = max(cv_rows, key=lambda row: row[1])[0]
    return best_k, cv_rows


def prediction_cost(y, predicted):
    metrics = confusion_metrics(y, predicted)
    return 50 * metrics["fp"] + 400 * metrics["fn"]


def choose_cost_threshold(y_train, training_probabilities):
    """Choose on training rows only, retaining the smaller threshold on ties."""
    candidates = [
        (k / 20, prediction_cost(y_train, training_probabilities >= k / 20))
        for k in range(1, 20)
    ]
    return min(candidates, key=lambda row: row[1])
