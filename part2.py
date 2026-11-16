"""Train and evaluate churn classifiers with the prescribed split and folds."""

import numpy as np

from common import load_data, write_csv, write_results
from models import (
    binary_cross_entropy,
    choose_cost_threshold,
    confusion_metrics,
    fit_logistic,
    knn_probability,
    predict_probability,
    prediction_cost,
    select_knn_k,
    split_and_standardise,
)


def run():
    X, y, _ = load_data()
    split = split_and_standardise(X, y)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    results = {
        "train_rows": len(y_train),
        "train_positives": int(y_train.sum()),
        "test_rows": len(y_test),
        "test_positives": int(y_test.sum()),
    }

    weights, bias, losses = fit_logistic(X_train, y_train)
    for epoch in (0, 1, 10, 100, 1000):
        results[f"loss_epoch_{epoch}"] = losses[epoch]
    results.update({f"weight_X{j + 1}": w for j, w in enumerate(weights)})
    results["bias"] = bias
    train_probabilities = predict_probability(X_train, weights, bias)
    test_probabilities = predict_probability(X_test, weights, bias)
    results["test_bce"] = binary_cross_entropy(y_test, test_probabilities)

    for rate in (0.01, 0.1, 1.0):
        _, _, rate_losses = fit_logistic(
            X_train, y_train, learning_rate=rate, updates=200
        )
        results[f"loss_epoch_200_eta_{rate}"] = rate_losses[200]
    results.update(
        {f"odds_ratio_X{j + 1}": np.exp(w) for j, w in enumerate(weights)}
    )

    reported_metrics = (
        "tp", "fp", "tn", "fn", "accuracy",
        "precision", "recall", "f1", "specificity",
    )
    for threshold in (0.5, 0.3):
        metrics = confusion_metrics(y_test, test_probabilities >= threshold)
        for key in reported_metrics:
            results[f"tau_{threshold}_{key}"] = metrics[key]
    results["blind_accuracy"] = confusion_metrics(
        y_test, np.zeros(len(y_test), dtype=int)
    )["accuracy"]

    threshold_rows = []
    for k in range(1, 10):
        threshold = k / 10
        metrics = confusion_metrics(y_test, test_probabilities >= threshold)
        threshold_rows.append(
            [threshold] + [metrics[key] for key in ("tp", "fp", "tn", "fn", "tpr", "fpr")]
        )
    write_csv(
        "part2_thresholds.csv",
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        threshold_rows,
    )

    best_k, cv_rows = select_knn_k(X_train, y_train)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)
    results["knn_best_k"] = best_k
    neighbour_share = knn_probability(X_train, y_train, X_test, best_k)
    knn_metrics = confusion_metrics(y_test, neighbour_share >= 0.5)
    for key in reported_metrics[:-1]:
        results[f"knn_{key}"] = knn_metrics[key]

    threshold, training_cost = choose_cost_threshold(y_train, train_probabilities)
    results["cost_tau"] = threshold
    results["cost_train_at_tau"] = training_cost
    results["cost_test_at_tau"] = prediction_cost(
        y_test, test_probabilities >= threshold
    )
    results["cost_test_at_0.5"] = prediction_cost(
        y_test, test_probabilities >= 0.5
    )
    results["cost_test_flag_nobody"] = prediction_cost(
        y_test, np.zeros(len(y_test), dtype=int)
    )
    write_results("part2_results.txt", results)
    return results


if __name__ == "__main__":
    run()
