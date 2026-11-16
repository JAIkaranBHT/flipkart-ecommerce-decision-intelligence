"""Split gains, ensemble rules, two AdaBoost rounds and network sizing.

This entry point deliberately refits the shared Part 2 implementations rather
than reading rounded probabilities or metrics from an earlier output file.
"""

import math

import numpy as np

from common import load_data, write_csv, write_results
from models import (
    confusion_metrics,
    fit_logistic,
    knn_probability,
    predict_probability,
    select_knn_k,
    split_and_standardise,
)


def split_conditions(raw_features):
    """Return each pinned split's yes-side mask, using original feature units."""
    return {
        "S1": raw_features[:, 5] == 1,
        "S2": raw_features[:, 6] == 1,
        "S3": raw_features[:, 2] > 0.5,
        "S4": raw_features[:, 0] <= 24,
    }


def node_impurities(labels):
    """Calculate binary entropy and Gini directly from the class proportions."""
    if len(labels) == 0:
        return 0.0, 0.0
    positive_rate = float(np.sum(labels)) / len(labels)
    proportions = (positive_rate, 1.0 - positive_rate)
    entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
    gini = 1.0 - sum(p * p for p in proportions)
    return entropy, gini


def evaluate_splits(raw_train, y_train):
    """Calculate the root impurities, four gains and two best split names."""
    root_entropy, root_gini = node_impurities(y_train)
    rows = []
    for name, yes in split_conditions(raw_train).items():
        no = ~yes
        n_yes = int(np.sum(yes))
        n_no = int(np.sum(no))
        yes_entropy, yes_gini = node_impurities(y_train[yes])
        no_entropy, no_gini = node_impurities(y_train[no])
        yes_share = n_yes / len(y_train)
        no_share = n_no / len(y_train)
        information_gain = (
            root_entropy - yes_share * yes_entropy - no_share * no_entropy
        )
        gini_gain = root_gini - yes_share * yes_gini - no_share * no_gini
        rows.append(
            (
                name,
                n_yes,
                int(np.sum(y_train[yes])),
                n_no,
                int(np.sum(y_train[no])),
                information_gain,
                gini_gain,
            )
        )
    # max retains the first candidate if gains tie, so split order is stable.
    best_information = max(rows, key=lambda row: row[5])[0]
    best_gini = max(rows, key=lambda row: row[6])[0]
    results = {
        "root_entropy": root_entropy,
        "root_gini": root_gini,
        "best_split_by_info_gain": best_information,
        "best_split_by_gini_gain": best_gini,
    }
    return results, rows


def evaluate_voting(split, best_split):
    """Fit the three base learners on training rows and evaluate five rules."""
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    weights, bias, _ = fit_logistic(X_train, y_train)
    logistic_probability = predict_probability(X_test, weights, bias)
    best_k, _ = select_knn_k(X_train, y_train)
    neighbour_probability = knn_probability(
        X_train, y_train, X_test, best_k
    )

    train_yes = split_conditions(split["raw_train"])[best_split]
    test_yes = split_conditions(split["raw_test"])[best_split]
    # The pinned candidates all have nonempty sides in the training dataset.
    rate_yes = float(np.mean(y_train[train_yes]))
    rate_no = float(np.mean(y_train[~train_yes]))
    stump_probability = np.where(test_yes, rate_yes, rate_no)

    logistic_prediction = logistic_probability >= 0.3
    # Odd K makes probability >= 0.5 exactly the specified majority rule.
    knn_prediction = neighbour_probability >= 0.5
    stump_prediction = stump_probability >= 0.3
    vote_count = (
        logistic_prediction.astype(int)
        + knn_prediction.astype(int)
        + stump_prediction.astype(int)
    )
    average_probability = (
        logistic_probability + neighbour_probability + stump_probability
    ) / 3.0
    predictions = (
        ("logistic", logistic_prediction),
        ("knn", knn_prediction),
        ("stump", stump_prediction),
        ("hard_vote", vote_count >= 2),
        ("soft_vote", average_probability >= 0.3),
    )
    rows = []
    for name, prediction in predictions:
        metrics = confusion_metrics(y_test, prediction)
        rows.append(
            (
                name,
                metrics["tp"],
                metrics["fp"],
                metrics["tn"],
                metrics["fn"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )
        )
    return {"stump_rate_yes": rate_yes, "stump_rate_no": rate_no}, rows


def two_round_adaboost():
    """Apply the prescribed weak rules in sequence, retaining every weight."""
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=int)
    weak_rules = (
        np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=int),
        np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=int),
    )
    weights = np.full(len(y), 1.0 / len(y))
    ensemble_score = np.zeros(len(y), dtype=float)
    results = {}
    rows = []
    for round_number, rule in enumerate(weak_rules, start=1):
        error = float(np.sum(weights[rule != y]))
        alpha = 0.5 * math.log((1.0 - error) / error)
        unnormalised = weights * np.exp(-alpha * y * rule)
        after = unnormalised / np.sum(unnormalised)
        results[f"round{round_number}_error"] = error
        results[f"round{round_number}_alpha"] = alpha
        for index in range(len(y)):
            rows.append(
                (
                    round_number,
                    index + 1,
                    weights[index],
                    unnormalised[index],
                    after[index],
                )
            )
        ensemble_score += alpha * rule
        weights = after
    results["ensemble_accuracy"] = float(np.mean(np.sign(ensemble_score) == y))
    return results, rows


def spatial_output(size, kernel, stride, padding):
    """Compute one spatial side from the supplied convolution/pooling rule."""
    numerator = size - kernel + 2 * padding
    if numerator < 0 or numerator % stride != 0:
        raise ValueError("The pinned architecture must have integral layer sizes.")
    return numerator // stride + 1


def dense_parameters(inputs, outputs):
    """One weight per input-output pair and one bias per output neuron."""
    return (inputs + 1) * outputs


def convolution_parameters(kernel, input_channels, output_channels):
    """Each filter has kernel squared times input depth weights plus one bias."""
    return (kernel * kernel * input_channels + 1) * output_channels


def architecture_details():
    """Return required counts and a complete shape table for the README.

    Dense outputs are shown as 1 x 1 x width; flatten changes representation
    without adding parameters. No numerical learning is performed.
    """
    image_side, image_channels, image_classes = 28, 1, 10
    mlp_widths = (image_side * image_side * image_channels, 16, 16, image_classes)
    mlp_names = ("mlp_hidden1", "mlp_hidden2", "mlp_output")
    rows = [
        (name, dense_parameters(inputs, outputs))
        for name, inputs, outputs in zip(
            mlp_names, mlp_widths[:-1], mlp_widths[1:]
        )
    ]

    conv1_channels, conv2_channels = 6, 16
    conv1_side = spatial_output(image_side, kernel=5, stride=1, padding=2)
    pool1_side = spatial_output(conv1_side, kernel=2, stride=2, padding=0)
    conv2_side = spatial_output(pool1_side, kernel=5, stride=1, padding=0)
    pool2_side = spatial_output(conv2_side, kernel=2, stride=2, padding=0)
    flattened_width = pool2_side * pool2_side * conv2_channels
    dense_width = 32
    conv1_count = convolution_parameters(5, image_channels, conv1_channels)
    conv2_count = convolution_parameters(5, conv1_channels, conv2_channels)
    dense1_count = dense_parameters(flattened_width, dense_width)
    dense2_count = dense_parameters(dense_width, image_classes)
    rows.extend(
        [
            ("cnn_conv1", conv1_count),
            ("cnn_conv2", conv2_count),
            ("cnn_dense1", dense1_count),
            ("cnn_dense2", dense2_count),
        ]
    )
    shape_rows = [
        ("input", (image_side, image_side, image_channels), 0),
        ("conv1", (conv1_side, conv1_side, conv1_channels), conv1_count),
        ("relu1", (conv1_side, conv1_side, conv1_channels), 0),
        ("pool1", (pool1_side, pool1_side, conv1_channels), 0),
        ("conv2", (conv2_side, conv2_side, conv2_channels), conv2_count),
        ("relu2", (conv2_side, conv2_side, conv2_channels), 0),
        ("pool2", (pool2_side, pool2_side, conv2_channels), 0),
        ("flatten", (1, 1, flattened_width), 0),
        ("dense1", (1, 1, dense_width), dense1_count),
        ("relu3", (1, 1, dense_width), 0),
        ("dense2", (1, 1, image_classes), dense2_count),
    ]

    large_side, large_channels, fc_width = 128, 3, 1024
    token_width, hidden_width, topic_count = 64, 128, 16
    rnn_xh = hidden_width * token_width
    rnn_hh = hidden_width * hidden_width
    rnn_hy = topic_count * hidden_width
    rnn_bias_h, rnn_bias_y = hidden_width, topic_count
    results = {
        "mlp_total": sum(count for name, count in rows if name.startswith("mlp_")),
        "cnn_total": sum(count for name, count in rows if name.startswith("cnn_")),
        "fc_layer_params": dense_parameters(
            large_side * large_side * large_channels, fc_width
        ),
        "conv_layer_params": convolution_parameters(3, large_channels, 64),
        "rnn_total": rnn_xh + rnn_hh + rnn_hy + rnn_bias_h + rnn_bias_y,
    }
    return results, rows, shape_rows


def main():
    X, y, _ = load_data()
    split = split_and_standardise(X, y)
    results, split_rows = evaluate_splits(split["raw_train"], split["y_train"])
    stump_results, voting_rows = evaluate_voting(
        split, results["best_split_by_info_gain"]
    )
    results.update(stump_results)
    boosting_results, boosting_rows = two_round_adaboost()
    results.update(boosting_results)
    training_rows = len(split["y_train"])
    results["oob_fraction"] = (1.0 - 1.0 / training_rows) ** training_rows
    architecture_results, architecture_rows, _ = architecture_details()
    results.update(architecture_results)

    write_results("part4_results.txt", results)
    write_csv(
        "part4_splits.csv",
        [
            "split",
            "n_yes",
            "positives_yes",
            "n_no",
            "positives_no",
            "info_gain",
            "gini_gain",
        ],
        split_rows,
    )
    write_csv(
        "part4_voting.csv",
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )
    write_csv(
        "part4_adaboost.csv",
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        boosting_rows,
    )
    write_csv("part4_architecture.csv", ["layer", "parameters"], architecture_rows)


if __name__ == "__main__":
    main()
