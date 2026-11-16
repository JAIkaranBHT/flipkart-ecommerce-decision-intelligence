"""Offline verification: python -m unittest -v test_project.

Only the standard library and NumPy are required. Entry points write into a
temporary output directory during testing; committed outputs are never edited.
Reference constants below are assertions, never production inputs.
"""

import contextlib
import csv
import hashlib
import io
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import common
import models
import part1
import part2
import part3
import part4


ROOT = Path(__file__).resolve().parent
FLOAT_PATTERN = re.compile(r"-?\d+\.\d{4}")
INTEGER_PATTERN = re.compile(r"-?\d+")

PART1_KEYS = set(
    """n mean_value sd_value se_value ci95_low ci95_high z_stat reject_h0
    n_required reg_slope reg_intercept tss ess rss tss_equals_ess_plus_rss
    r2 rmse mae gd_J0 gd_J1 gd_J10 gd_J100 gd_J500 gd_slope gd_intercept
    gd_slope_gap resid_sd_lower_raw resid_sd_upper_raw resid_sd_ratio_raw
    loglog_intercept loglog_slope loglog_r2 loglog_rmse resid_sd_lower_log
    resid_sd_upper_log resid_sd_ratio_log mr_r2 mr_adj_r2 loglog_adj_r2""".split()
) | {f"mr_beta_{j}" for j in range(8)}
PART2_KEYS = set(
    """train_rows train_positives test_rows test_positives bias test_bce
    blind_accuracy knn_best_k cost_tau cost_train_at_tau cost_test_at_tau
    cost_test_at_0.5 cost_test_flag_nobody""".split()
)
PART2_KEYS |= {f"loss_epoch_{j}" for j in (0, 1, 10, 100, 1000)}
PART2_KEYS |= {f"loss_epoch_200_eta_{rate}" for rate in ("0.01", "0.1", "1.0")}
PART2_KEYS |= {
    f"{prefix}_X{j}" for prefix in ("weight", "odds_ratio") for j in range(1, 8)
}
REPORT_METRICS = ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1")
PART2_KEYS |= {
    f"tau_{tau}_{key}"
    for tau in ("0.5", "0.3")
    for key in (*REPORT_METRICS, "specificity")
}
PART2_KEYS |= {f"knn_{key}" for key in REPORT_METRICS}
PART3_KEYS = {
    "elbow_k", "eigenvalue_sum", "components_for_80pct",
    "single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut",
}
PART3_KEYS |= {f"wcss_k{k}" for k in range(2, 7)}
PART3_KEYS |= {
    f"{prefix}_{j}"
    for prefix in ("eigenvalue", "explained_ratio", "cumulative_ratio")
    for j in range(1, 6)
}
PART3_KEYS |= {
    f"pc{pc}_loading_X{j}" for pc in (1, 2) for j in range(1, 6)
}
PART3_KEYS |= {f"row{row}_pc{pc}" for row in (1, 2, 3) for pc in (1, 2)}
PART3_KEYS |= {
    f"{prefix}_segment_{segment}"
    for prefix in ("size", "mean_pc1")
    for segment in "ABC"
}
PART4_KEYS = set(
    """root_entropy root_gini best_split_by_info_gain best_split_by_gini_gain
    stump_rate_yes stump_rate_no round1_error round1_alpha round2_error
    round2_alpha ensemble_accuracy oob_fraction mlp_total cnn_total
    fc_layer_params conv_layer_params rnn_total""".split()
)
RESULT_KEYS = {1: PART1_KEYS, 2: PART2_KEYS, 3: PART3_KEYS, 4: PART4_KEYS}
INTEGER_KEYS = {
    "n", "n_required", "train_rows", "train_positives", "test_rows",
    "test_positives", "knn_best_k", "cost_train_at_tau", "cost_test_at_tau",
    "cost_test_at_0.5", "cost_test_flag_nobody", "elbow_k",
    "components_for_80pct", "single_clusters_at_cut", "complete_clusters_at_cut",
    "average_clusters_at_cut", "mlp_total", "cnn_total", "fc_layer_params",
    "conv_layer_params", "rnn_total",
}
INTEGER_KEYS |= {f"size_segment_{s}" for s in "ABC"}
INTEGER_KEYS |= {
    f"{prefix}_{cell}"
    for prefix in ("tau_0.5", "tau_0.3", "knn")
    for cell in ("tp", "fp", "tn", "fn")
}
BOOLEAN_KEYS = {"reject_h0", "tss_equals_ess_plus_rss"}
TEXT_KEYS = {"best_split_by_info_gain", "best_split_by_gini_gain"}
# filename: (exact header, data-row count, integer columns, text columns)
CSV_SCHEMAS = {
    "part2_thresholds.csv": (
        "tau,tp,fp,tn,fn,tpr,fpr", 9, {"tp", "fp", "tn", "fn"}, set()
    ),
    "part2_knn_cv.csv": ("K,mean_cv_f1", 11, {"K"}, set()),
    "part3_segments.csv": (
        "segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value",
        3, {"size"}, {"segment"},
    ),
    "part3_hierarchical.csv": (
        "linkage,merge_no,cluster_a,cluster_b,height",
        15, {"merge_no"}, {"linkage", "cluster_a", "cluster_b"},
    ),
    "part4_splits.csv": (
        "split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain",
        4, {"n_yes", "positives_yes", "n_no", "positives_no"}, {"split"},
    ),
    "part4_voting.csv": (
        "model,tp,fp,tn,fn,precision,recall,f1",
        5, {"tp", "fp", "tn", "fn"}, {"model"},
    ),
    "part4_adaboost.csv": (
        "round,point,weight_before,weight_unnormalised,weight_after",
        16, {"round", "point"}, set(),
    ),
    "part4_architecture.csv": ("layer,parameters", 7, {"parameters"}, {"layer"}),
}


def read_results(path):
    pairs = [line.split(" = ") for line in path.read_text().splitlines()]
    if any(len(pair) != 2 for pair in pairs):
        raise AssertionError(f"Malformed key = value line in {path.name}")
    if len({pair[0] for pair in pairs}) != len(pairs):
        raise AssertionError(f"Duplicate result key in {path.name}")
    return dict(pairs)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def independent_metrics(actual, predicted):
    """Count directly and use the equivalent F1 count formula."""
    counts = {(a, p): 0 for a in (0, 1) for p in (0, 1)}
    for actual_value, prediction in zip(actual, predicted):
        counts[int(actual_value), int(prediction)] += 1
    tp, fp = counts[1, 1], counts[0, 1]
    tn, fn = counts[0, 0], counts[1, 0]
    ratio = lambda a, b: a / b if b else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": ratio(tn, tn + fp),
        "tpr": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
    }


class DataAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.value = common.load_data()
        cls.split = models.split_and_standardise(cls.X, cls.y)

    def test_generator_verbatim_and_dataset_hash(self):
        self.assertEqual(
            hashlib.sha256((ROOT / "generate_data.py").read_bytes()).hexdigest(),
            "1578e51eb70624bf9a897ecc3dc20d6584d6e38ac4ffacfead1149a46de8dd63",
        )
        self.assertEqual(
            hashlib.md5((ROOT / "data/records.csv").read_bytes()).hexdigest(),
            "c6f249ba693f1ebeed318d6d5e18de23",
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "generate_data.py")],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            self.assertEqual(
                completed.stdout.strip(),
                "800 10 170 c6f249ba693f1ebeed318d6d5e18de23",
            )
            self.assertEqual(
                (Path(directory) / "data/records.csv").read_bytes(),
                (ROOT / "data/records.csv").read_bytes(),
            )

    def test_data_columns_and_pinned_split(self):
        with (ROOT / "data/records.csv").open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(
            rows[0],
            ["record_id", *common.FEATURE_NAMES, "churned", "order_value_inr_hundreds"],
        )
        self.assertEqual([row[0] for row in rows[1:]], [f"E{i:04d}" for i in range(1, 801)])
        expected_train = sorted(
            i for label in (0, 1)
            for i in [j for j, y in enumerate(self.y) if y == label][
                :7 * int(np.count_nonzero(self.y == label)) // 10
            ]
        )
        np.testing.assert_array_equal(self.split["train_indices"], expected_train)
        np.testing.assert_array_equal(
            np.sort(np.r_[self.split["train_indices"], self.split["test_indices"]]),
            np.arange(800),
        )
        self.assertEqual(len(expected_train), 560)
        self.assertEqual(int(self.split["y_train"].sum()), 119)
        self.assertEqual(len(self.split["y_test"]), 240)
        self.assertEqual(int(self.split["y_test"].sum()), 51)

    def test_training_only_sample_standardisation(self):
        train = self.X[self.split["train_indices"]]
        np.testing.assert_allclose(self.split["train_mean"], train.mean(axis=0))
        np.testing.assert_allclose(self.split["train_sd"], train.std(axis=0, ddof=1))
        np.testing.assert_allclose(self.split["X_train"].mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(self.split["X_train"].std(axis=0, ddof=1), 1)
        perturbed = self.X.copy()
        perturbed[self.split["test_indices"]] = 1e8
        changed = models.split_and_standardise(perturbed, self.y)
        for key in ("train_mean", "train_sd", "X_train", "y_train"):
            np.testing.assert_array_equal(self.split[key], changed[key])
        expected_test = (
            self.X[self.split["test_indices"]] - train.mean(axis=0)
        ) / train.std(axis=0, ddof=1)
        np.testing.assert_allclose(self.split["X_test"], expected_test)

    def test_logistic_full_batch_updates_and_losses(self):
        X, y = self.split["X_train"], self.split["y_train"]
        weights, bias, history = models.fit_logistic(X, y)
        design = np.column_stack((X, np.ones(len(y))))
        beta = np.zeros(design.shape[1])
        for update in range(1001):
            scores = design @ beta
            if update in (0, 1, 10, 100, 1000):
                expected_loss = np.mean(np.logaddexp(0, scores) - y * scores)
                self.assertAlmostEqual(history[update], expected_loss, places=13)
            if update < 1000:
                beta -= 0.1 * (design.T @ (1 / (1 + np.exp(-scores)) - y)) / len(y)
        np.testing.assert_allclose(np.r_[weights, bias], beta, rtol=1e-12, atol=1e-12)
        self.assertTrue(all(history[i] >= history[i + 1] for i in range(1000)))
        self.assertAlmostEqual(history[0], math.log(2))

    def test_metrics_degenerate_denominators_and_probability_boundaries(self):
        cases = [
            ([], []), ([0, 0], [0, 0]), ([1, 1], [0, 0]),
            ([1, 1], [1, 1]), ([0, 0], [1, 1]),
            ([0, 1, 0, 1], [1, 1, 0, 0]),
        ]
        for actual, predicted in cases:
            with self.subTest(actual=actual, predicted=predicted):
                self.assertEqual(
                    models.confusion_metrics(actual, predicted),
                    independent_metrics(actual, predicted),
                )
        probability = models.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_array_equal(probability, [0.0, 0.5, 1.0])
        self.assertTrue(math.isfinite(models.binary_cross_entropy(np.array([1, 0]), np.array([0., 1.]))))
        self.assertEqual(models.confusion_metrics([1], np.array([0.3]) >= 0.3)["tp"], 1)

    def test_stable_knn_distance_ties(self):
        reference = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        labels = np.array([1, 0, 0, 1])
        query = np.array([[0.0]])
        np.testing.assert_array_equal(
            models.neighbour_labels(reference, labels, query), [labels]
        )
        np.testing.assert_array_equal(
            models.knn_probability(reference, labels, query, 1), [1.0]
        )
        np.testing.assert_array_equal(
            models.knn_probability(reference, labels, query, 3), [1 / 3]
        )

    def test_pinned_knn_folds_mean_f1_and_smaller_k_tie(self):
        X, y = self.split["X_train"], self.split["y_train"]
        expected = {k: [] for k in range(1, 22, 2)}
        for fold in range(5):
            held = np.arange(len(y)) % 5 == fold
            reference_indices = np.flatnonzero(~held)
            distance = ((X[held, None] - X[None, ~held]) ** 2).sum(axis=2)
            # Explicit secondary key verifies the file-order distance tie rule.
            order = np.lexsort(
                (np.broadcast_to(reference_indices, distance.shape), distance), axis=1
            )
            ranked = y[reference_indices][order]
            for k in expected:
                predictions = ranked[:, :k].sum(axis=1) >= (k + 1) / 2
                expected[k].append(independent_metrics(y[held], predictions)["f1"])
        best, rows = models.select_knn_k(X, y)
        expected_means = [(k, sum(values) / 5) for k, values in expected.items()]
        np.testing.assert_allclose(rows, expected_means, atol=1e-14)
        self.assertEqual(best, max(expected_means, key=lambda row: row[1])[0])
        self.assertEqual(best, 5)
        tied_k, tied_rows = models.select_knn_k(np.arange(30.0)[:, None], np.zeros(30, int))
        self.assertEqual(tied_k, 1)
        self.assertTrue(all(score == 0 for _, score in tied_rows))

    def test_cost_threshold_training_grid_and_tie_break(self):
        self.assertEqual(
            models.choose_cost_threshold(np.array([0, 1]), np.array([0.5, 0.5])),
            (0.05, 50),
        )
        self.assertEqual(
            models.choose_cost_threshold(np.array([1]), np.array([0.05])),
            (0.05, 0),
        )
        X, y = self.split["X_train"], self.split["y_train"]
        w, b, _ = models.fit_logistic(X, y)
        p = models.predict_probability(X, w, b)
        candidates = []
        for k in range(1, 20):
            metrics = independent_metrics(y, p >= k / 20)
            candidates.append((50 * metrics["fp"] + 400 * metrics["fn"], k / 20))
        cost, tau = min(candidates)
        self.assertEqual(models.choose_cost_threshold(y, p), (tau, cost))
        self.assertEqual(tau, 0.15)

    def test_module_imports_do_not_train_or_write_outputs(self):
        script = """
import sys
from pathlib import Path
from unittest import mock
sys.path.insert(0, sys.argv[1])
import common
import models
common.OUTPUTS = Path(sys.argv[2])
with (
    mock.patch.object(common, "load_data", side_effect=AssertionError("import loaded data")),
    mock.patch.object(models, "fit_logistic", side_effect=AssertionError("import trained")),
):
    import generate_data
    import part1
    import part2
    import part3
    import part4
assert not list(common.OUTPUTS.iterdir()), "import wrote output files"
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(ROOT), directory],
                cwd=directory, check=True, capture_output=True, text=True,
            )
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


class NumericalConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.value = common.load_data()
        cls.standardised, _, _ = part3.standardise(cls.X[:, :5])

    def test_part1_estimates_and_normal_equation(self):
        result = part1.compute_results(self.X, self.value)
        self.assertEqual(set(result), PART1_KEYS)
        sd = math.sqrt(sum((v - self.value.mean()) ** 2 for v in self.value) / 799)
        self.assertAlmostEqual(result["sd_value"], sd)
        self.assertAlmostEqual(result["se_value"], sd / math.sqrt(800))
        self.assertEqual(result["n_required"], math.ceil((1.96 * sd / 0.5) ** 2))
        self.assertAlmostEqual(result["z_stat"], (self.value.mean() - 8) / result["se_value"])
        self.assertEqual(result["reject_h0"], abs(result["z_stat"]) > 1.96)
        self.assertAlmostEqual(result["mean_value"], 8.9382, places=4)
        self.assertAlmostEqual(result["tss"], result["ess"] + result["rss"], places=8)
        self.assertAlmostEqual(result["rmse"] ** 2, result["rss"] / 800)
        design = np.column_stack((
            np.ones(800), np.log(self.X[:, 3]), self.X[:, :3],
            self.X[:, 4:7],
        ))
        beta = np.array([result[f"mr_beta_{i}"] for i in range(8)])
        residual = np.log(self.value) - design @ beta
        np.testing.assert_allclose(design.T @ residual, 0, atol=1e-8)
        self.assertAlmostEqual(
            result["mr_adj_r2"], 1 - (1 - result["mr_r2"]) * 799 / 792
        )
        self.assertAlmostEqual(
            result["loglog_adj_r2"], 1 - (1 - result["loglog_r2"]) * 799 / 798
        )

    def test_gradient_descent_costs_by_closed_form_recurrence(self):
        x, y = self.X[:, 3], self.value
        z = (x - x.mean()) / x.std(ddof=1)
        result = part1.fit_line_by_gradient_descent(x, y)
        optimal_m = (z @ y) / (z @ z)
        for updates in (0, 1, 10, 100, 500):
            c = y.mean() * (1 - 0.8 ** updates)
            m = optimal_m * (1 - (1 - 0.2 * 799 / 800) ** updates)
            self.assertAlmostEqual(result[f"gd_J{updates}"], np.mean((y - m * z - c) ** 2), places=10)
        exact = part1.fit_simple_line(x, y)
        self.assertLess(abs(result["gd_slope"] - exact.slope), 1e-12)
        self.assertAlmostEqual(result["gd_intercept"], exact.intercept)

    def test_residual_halves_stable_ties_and_logged_scale(self):
        # A tied fitted value straddles the half boundary in a nontrivial order.
        x = np.array([0., 1., 1., 1., 1., 2.])
        y = np.array([0., -7., 9., 12., -5., 15.])
        fit = part1.fit_simple_line(x, y)
        order = sorted(range(len(x)), key=lambda i: (fit.fitted[i], i))
        expected = [fit.residuals[order[:3]].std(ddof=1), fit.residuals[order[3:]].std(ddof=1)]
        actual = part1.residual_spread_by_fitted_half(fit)
        np.testing.assert_allclose(actual, [*expected, expected[1] / expected[0]])
        result = part1.compute_results(self.X, self.value)
        for prefix, predictor, target in (
            ("raw", self.X[:, 3], self.value),
            ("log", np.log(self.X[:, 3]), np.log(self.value)),
        ):
            fit = part1.fit_simple_line(predictor, target)
            order = np.lexsort((np.arange(800), fit.fitted))
            lower = fit.residuals[order[:400]].std(ddof=1)
            upper = fit.residuals[order[400:]].std(ddof=1)
            self.assertAlmostEqual(result[f"resid_sd_ratio_{prefix}"], upper / lower)
        self.assertAlmostEqual(result["resid_sd_ratio_raw"], 3.7525, places=4)
        self.assertAlmostEqual(result["resid_sd_ratio_log"], 1.0396, places=4)

    def test_kmeans_pinned_starts_final_passes_and_centroids(self):
        expected_passes = {4: 21, 5: 15, 6: 27}
        wcss = {}
        for k in range(2, 7):
            labels, centroids, wcss[k], passes = part3.kmeans(self.standardised, k)
            distance = ((self.standardised[:, None] - centroids) ** 2).sum(axis=2)
            np.testing.assert_array_equal(labels, np.argmin(distance, axis=1))
            self.assertAlmostEqual(wcss[k], sum(distance[i, label] for i, label in enumerate(labels)))
            for cluster in range(k):
                np.testing.assert_allclose(
                    centroids[cluster], self.standardised[labels == cluster].mean(axis=0)
                )
            if k in expected_passes:
                self.assertEqual(passes, expected_passes[k])
        self.assertAlmostEqual(wcss[2], 2262.6753, places=4)
        self.assertEqual(part3.choose_elbow(wcss), 3)

    def test_kmeans_empty_clusters_and_centroid_ties(self):
        labels, centroids, wcss, passes = part3.kmeans(np.zeros((6, 2)), 3)
        np.testing.assert_array_equal(labels, np.zeros(6, dtype=int))
        np.testing.assert_array_equal(centroids, np.zeros((3, 2)))
        self.assertEqual(wcss, 0)
        self.assertEqual(passes, 2)
        labels, _, _, _ = part3.kmeans(np.array([[-1.], [0.], [1.], [0.]]), 2)
        self.assertEqual(labels[1], 0)
        self.assertEqual(labels[3], 0)

    def test_hierarchical_member_pairs_and_largest_gap_cut(self):
        expected = {
            "single": ([1.3, 1.4, 2.8, 2.9, 3.4], 4),
            "complete": ([1.3, 2.0, 2.8, 2.9, 7.9], 2),
            "average": ([1.3, 1.7, 2.8, 2.9, 50.9 / 9], 2),
        }
        for linkage, (heights, count) in expected.items():
            rows, cut_count, partition = part3.agglomerative(part3.COHORT_DISTANCES, linkage)
            np.testing.assert_allclose([row[-1] for row in rows], heights)
            self.assertEqual(cut_count, count)
            self.assertEqual(len(partition), count)
            self.assertEqual(rows[0][2:4], ("E3", "E6"))
            self.assertEqual(rows[1][2:4], ("E2", "E3+E6"))
            for row in rows:
                self.assertLess(int(row[2][1:].split("+")[0]), int(row[3][1:].split("+")[0]))
        _, _, single = part3.agglomerative(part3.COHORT_DISTANCES, "single")
        self.assertEqual(single, [(0,), (1, 2, 5), (3,), (4,)])
        _, _, complete = part3.agglomerative(part3.COHORT_DISTANCES, "complete")
        self.assertEqual(complete, [(0, 3, 4), (1, 2, 5)])

    def test_pca_seven_steps_signs_eigenpairs_and_projection(self):
        pca = part3.pca_seven_steps(self.standardised)
        eigenvalues, vectors = pca["eigenvalues"], pca["eigenvectors"]
        np.testing.assert_allclose(pca["mean_vector"], 0, atol=1e-14)
        np.testing.assert_allclose(np.diag(pca["covariance"]), 1)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(5), atol=1e-14)
        np.testing.assert_allclose(pca["covariance"] @ vectors, vectors * eigenvalues, atol=1e-14)
        self.assertTrue(np.all(np.diff(eigenvalues) < 0))
        self.assertAlmostEqual(eigenvalues.sum(), 5.0)
        for vector in vectors.T:
            self.assertGreater(vector[np.argmax(abs(vector))], 0)
        np.testing.assert_allclose(pca["scores"], pca["centred"] @ vectors[:, :2])
        np.testing.assert_allclose(pca["explained"], eigenvalues / 5)
        np.testing.assert_allclose(pca["cumulative"], np.cumsum(eigenvalues / 5))
        self.assertEqual(pca["components_for_80pct"], 3)
        np.testing.assert_allclose(
            pca["scores"][:3], [[1.9008, -0.8537], [-0.3505, 1.9241], [-1.4382, 1.4312]],
            atol=0.00005,
        )

    def test_split_conditions_boundaries_and_impurities(self):
        raw = np.zeros((3, 7))
        raw[:, 0] = [24, 24.1, 23.9]
        raw[:, 2] = [0.5, 0.50001, 0.49999]
        raw[:, 5] = [1, 0, 1]
        raw[:, 6] = [0, 1, 0]
        masks = part4.split_conditions(raw)
        self.assertEqual({k: v.tolist() for k, v in masks.items()}, {
            "S1": [True, False, True], "S2": [False, True, False],
            "S3": [False, True, False], "S4": [True, False, True],
        })
        for labels in ([], [0, 0], [1, 1]):
            self.assertEqual(part4.node_impurities(np.array(labels)), (0.0, 0.0))
        self.assertEqual(part4.node_impurities(np.array([0, 1])), (1.0, 0.5))
        split = models.split_and_standardise(self.X, self.y)
        result, rows = part4.evaluate_splits(split["raw_train"], split["y_train"])
        p = 119 / 560
        self.assertAlmostEqual(result["root_entropy"], -p * math.log2(p) - (1 - p) * math.log2(1 - p))
        self.assertAlmostEqual(result["root_gini"], 2 * p * (1 - p))
        for row in rows:
            _, ny, py, nn, pn, gain_e, gain_g = row
            self.assertEqual(ny + nn, 560)
            self.assertEqual(py + pn, 119)
            child_e = child_g = 0
            for n, positives in ((ny, py), (nn, pn)):
                q = positives / n
                child_e += n / 560 * (-q * math.log2(q) - (1 - q) * math.log2(1 - q))
                child_g += n / 560 * 2 * q * (1 - q)
            self.assertAlmostEqual(gain_e, result["root_entropy"] - child_e)
            self.assertAlmostEqual(gain_g, result["root_gini"] - child_g)
        self.assertEqual(result["best_split_by_info_gain"], max(rows, key=lambda r: r[5])[0])
        self.assertEqual(result["best_split_by_gini_gain"], max(rows, key=lambda r: r[6])[0])

    def test_adaboost_each_weight_update_and_round_handoff(self):
        result, rows = part4.two_round_adaboost()
        y = [1, 1, 1, -1, -1, -1, 1, -1]
        rules = [[1, 1, -1, -1, -1, -1, -1, -1], [-1, 1, 1, -1, 1, -1, 1, -1]]
        previous = [1 / 8] * 8
        scores = [0.] * 8
        for round_number, rule in enumerate(rules, 1):
            block = rows[(round_number - 1) * 8:round_number * 8]
            error = sum(w for w, a, p in zip(previous, y, rule) if a != p)
            alpha = math.log((1 - error) / error) / 2
            unnormalised = [w * math.exp(-alpha * a * p) for w, a, p in zip(previous, y, rule)]
            after = [w / sum(unnormalised) for w in unnormalised]
            self.assertAlmostEqual(result[f"round{round_number}_error"], error)
            self.assertAlmostEqual(result[f"round{round_number}_alpha"], alpha)
            np.testing.assert_allclose([r[2:] for r in block], np.array([previous, unnormalised, after]).T)
            self.assertAlmostEqual(sum(after), 1)
            if round_number == 1:
                self.assertAlmostEqual(after[2] + after[6], 0.5)
            previous = after
            scores = [s + alpha * p for s, p in zip(scores, rule)]
        self.assertAlmostEqual(result["round1_error"], 0.25)
        self.assertAlmostEqual(result["round2_error"], 1 / 6)
        self.assertGreater(result["round2_alpha"], result["round1_alpha"])
        self.assertEqual(result["ensemble_accuracy"], sum((s > 0) == (a > 0) for s, a in zip(scores, y)) / 8)

    def test_network_shapes_biases_and_parameter_totals(self):
        result, rows, shapes = part4.architecture_details()
        self.assertEqual(dict(rows), {
            "mlp_hidden1": 12560, "mlp_hidden2": 272, "mlp_output": 170,
            "cnn_conv1": 156, "cnn_conv2": 2416, "cnn_dense1": 12832, "cnn_dense2": 330,
        })
        self.assertEqual(result, {
            "mlp_total": 13002, "cnn_total": 15734, "fc_layer_params": 50332672,
            "conv_layer_params": 1792, "rnn_total": 26768,
        })
        shape_map = {name: shape for name, shape, _ in shapes}
        self.assertEqual(shape_map["conv1"], (28, 28, 6))
        self.assertEqual(shape_map["pool1"], (14, 14, 6))
        self.assertEqual(shape_map["conv2"], (10, 10, 16))
        self.assertEqual(shape_map["pool2"], (5, 5, 16))
        self.assertEqual(shape_map["flatten"], (1, 1, 400))
        self.assertEqual(sum(count for _, _, count in shapes), result["cnn_total"])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.generated = Path(cls.directory.name)
        with (
            mock.patch.object(common, "OUTPUTS", cls.generated),
            mock.patch.object(part2, "select_knn_k", wraps=models.select_knn_k) as cv,
            mock.patch.object(part2, "choose_cost_threshold", wraps=models.choose_cost_threshold) as cost,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            part1.main()
            part2.run()
            part3.main()
            part4.main()
        cls.cv_inputs = cv.call_args.args
        cls.cost_inputs = cost.call_args.args
        cls.results = {
            n: read_results(cls.generated / f"part{n}_results.txt") for n in range(1, 5)
        }

    def assertPrintedEqual(self, actual, expected):
        self.assertEqual(actual, common.format_value(expected))

    def test_exact_result_key_sets_and_value_formats(self):
        for part, expected in RESULT_KEYS.items():
            result = self.results[part]
            self.assertEqual(set(result), expected, f"Part {part} key mismatch")
            for key, value in result.items():
                with self.subTest(part=part, key=key):
                    if key in INTEGER_KEYS:
                        self.assertRegex(value, rf"^{INTEGER_PATTERN.pattern}$")
                    elif key in BOOLEAN_KEYS:
                        self.assertIn(value, {"yes", "no"})
                    elif key in TEXT_KEYS:
                        self.assertIn(value, {"S1", "S2", "S3", "S4"})
                    else:
                        self.assertRegex(value, rf"^{FLOAT_PATTERN.pattern}$")

    def test_exact_csv_headers_rows_column_counts_and_four_dp(self):
        for filename, (header, row_count, integer_columns, text_columns) in CSV_SCHEMAS.items():
            with (self.generated / filename).open(newline="") as handle:
                raw_rows = list(csv.reader(handle))
            self.assertEqual(raw_rows[0], header.split(","), filename)
            self.assertEqual(len(raw_rows), row_count + 1, filename)
            for row in raw_rows[1:]:
                self.assertEqual(len(row), len(raw_rows[0]), filename)
                for key, value in zip(raw_rows[0], row):
                    with self.subTest(filename=filename, key=key):
                        if key in text_columns:
                            self.assertTrue(value)
                        elif key in integer_columns:
                            self.assertRegex(value, rf"^{INTEGER_PATTERN.pattern}$")
                        else:
                            self.assertRegex(value, rf"^{FLOAT_PATTERN.pattern}$")

    def test_committed_outputs_regenerate_byte_for_byte(self):
        filenames = {f"part{n}_results.txt" for n in range(1, 5)} | set(CSV_SCHEMAS)
        self.assertEqual({p.name for p in self.generated.iterdir()}, filenames)
        self.assertEqual(
            {p.name for p in (ROOT / "outputs").iterdir() if p.is_file()}, filenames
        )
        for filename in sorted(filenames):
            with self.subTest(filename=filename):
                self.assertEqual(
                    (ROOT / "outputs" / filename).read_bytes(),
                    (self.generated / filename).read_bytes(),
                )

    def test_part2_selection_receives_only_training_rows(self):
        X, y, _ = common.load_data()
        split = models.split_and_standardise(X, y)
        self.assertEqual(len(self.cv_inputs[0]), 560)
        np.testing.assert_array_equal(self.cv_inputs[0], split["X_train"])
        np.testing.assert_array_equal(self.cv_inputs[1], split["y_train"])
        np.testing.assert_array_equal(self.cost_inputs[0], split["y_train"])
        w, b, _ = models.fit_logistic(split["X_train"], split["y_train"])
        np.testing.assert_array_equal(
            self.cost_inputs[1], models.predict_probability(split["X_train"], w, b)
        )

    def test_classifier_metrics_threshold_sweep_and_cost_consistency(self):
        X, y, _ = common.load_data()
        split = models.split_and_standardise(X, y)
        w, b, _ = models.fit_logistic(split["X_train"], split["y_train"])
        p = models.predict_probability(split["X_test"], w, b)
        result = self.results[2]
        for tau in (0.5, 0.3):
            expected = independent_metrics(split["y_test"], p >= tau)
            for key in (*REPORT_METRICS, "specificity"):
                self.assertPrintedEqual(result[f"tau_{tau}_{key}"], expected[key])
        threshold_rows = csv_rows(self.generated / "part2_thresholds.csv")
        self.assertEqual([float(row["tau"]) for row in threshold_rows], [i / 10 for i in range(1, 10)])
        for row in threshold_rows:
            expected = independent_metrics(split["y_test"], p >= float(row["tau"]))
            for key in ("tp", "fp", "tn", "fn", "tpr", "fpr"):
                self.assertPrintedEqual(row[key], expected[key])
        cv = csv_rows(self.generated / "part2_knn_cv.csv")
        self.assertEqual([int(row["K"]) for row in cv], list(range(1, 22, 2)))
        tau = float(result["cost_tau"])
        for threshold, key in ((tau, "cost_test_at_tau"), (0.5, "cost_test_at_0.5")):
            expected = independent_metrics(split["y_test"], p >= threshold)
            self.assertEqual(int(result[key]), 50 * expected["fp"] + 400 * expected["fn"])
        self.assertEqual(int(result["cost_test_flag_nobody"]), 51 * 400)
        self.assertLess(float(self.results[1]["gd_slope_gap"]), 0.001)

    def test_segment_order_profiles_and_pca_reported_values(self):
        X, y, value = common.load_data()
        z, _, _ = part3.standardise(X[:, :5])
        labels, _, _, _ = part3.kmeans(z, 3)
        clusters = sorted(range(3), key=lambda cluster: X[labels == cluster, 2].mean())
        rows = csv_rows(self.generated / "part3_segments.csv")
        self.assertEqual([r["segment"] for r in rows], list("ABC"))
        self.assertEqual(sum(int(r["size"]) for r in rows), 800)
        pca = part3.pca_seven_steps(z)
        for segment, cluster, row in zip("ABC", clusters, rows):
            mask = labels == cluster
            self.assertPrintedEqual(row["size"], int(mask.sum()))
            self.assertEqual(self.results[3][f"size_segment_{segment}"], row["size"])
            for j in range(5):
                self.assertPrintedEqual(row[f"mean_X{j + 1}"], X[mask, j].mean())
            self.assertPrintedEqual(row["outcome_rate"], y[mask].mean())
            self.assertPrintedEqual(row["mean_value"], value[mask].mean())
            self.assertPrintedEqual(
                self.results[3][f"mean_pc1_segment_{segment}"], pca["scores"][mask, 0].mean()
            )
        for row in range(3):
            for pc in (1, 2):
                self.assertPrintedEqual(
                    self.results[3][f"row{row + 1}_pc{pc}"], pca["scores"][row, pc - 1]
                )

    def test_voting_rules_and_cross_part_reuse(self):
        X, y, _ = common.load_data()
        split = models.split_and_standardise(X, y)
        w, b, _ = models.fit_logistic(split["X_train"], split["y_train"])
        lp = models.predict_probability(split["X_test"], w, b)
        best_k = int(self.results[2]["knn_best_k"])
        kp = models.knn_probability(split["X_train"], split["y_train"], split["X_test"], best_k)
        best_split = self.results[4]["best_split_by_info_gain"]
        train_yes = part4.split_conditions(split["raw_train"])[best_split]
        test_yes = part4.split_conditions(split["raw_test"])[best_split]
        rate_yes = split["y_train"][train_yes].mean()
        rate_no = split["y_train"][~train_yes].mean()
        sp = np.where(test_yes, rate_yes, rate_no)
        predictions = [
            lp >= 0.3, kp >= 0.5, sp >= 0.3,
            np.sum([lp >= 0.3, kp >= 0.5, sp >= 0.3], axis=0) >= 2,
            (lp + kp + sp) / 3 >= 0.3,
        ]
        rows = csv_rows(self.generated / "part4_voting.csv")
        names = ["logistic", "knn", "stump", "hard_vote", "soft_vote"]
        self.assertEqual([row["model"] for row in rows], names)
        for row, prediction in zip(rows, predictions):
            expected = independent_metrics(split["y_test"], prediction)
            for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
                self.assertPrintedEqual(row[key], expected[key])
        for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
            self.assertEqual(rows[0][key], self.results[2][f"tau_0.3_{key}"])
            self.assertEqual(rows[1][key], self.results[2][f"knn_{key}"])
        self.assertPrintedEqual(self.results[4]["stump_rate_yes"], rate_yes)
        self.assertPrintedEqual(self.results[4]["stump_rate_no"], rate_no)
        self.assertPrintedEqual(self.results[4]["oob_fraction"], (559 / 560) ** 560)

    def test_csv_required_row_order_and_boost_weight_totals(self):
        rows = csv_rows(self.generated / "part3_hierarchical.csv")
        self.assertEqual(
            [(row["linkage"], int(row["merge_no"])) for row in rows],
            [(name, merge) for name in ("single", "complete", "average") for merge in range(1, 6)],
        )
        self.assertEqual(
            [r["split"] for r in csv_rows(self.generated / "part4_splits.csv")],
            ["S1", "S2", "S3", "S4"],
        )
        self.assertEqual(
            [r["layer"] for r in csv_rows(self.generated / "part4_architecture.csv")],
            ["mlp_hidden1", "mlp_hidden2", "mlp_output", "cnn_conv1", "cnn_conv2", "cnn_dense1", "cnn_dense2"],
        )
        boost = csv_rows(self.generated / "part4_adaboost.csv")
        self.assertEqual(
            [(int(r["round"]), int(r["point"])) for r in boost],
            [(round_no, point) for round_no in (1, 2) for point in range(1, 9)],
        )
        for start in (0, 8):
            # Eight independently rounded weights can accumulate 0.0004 error.
            self.assertAlmostEqual(
                sum(float(r["weight_after"]) for r in boost[start:start + 8]), 1, delta=0.0004
            )
        for before, after in zip(boost[8:], boost[:8]):
            self.assertEqual(before["weight_before"], after["weight_after"])

    def test_readme_first_line_heading_order_memo_and_self_containment(self):
        text = (ROOT / "README.md").read_text()
        self.assertEqual(text.splitlines()[0], "Domain: E-commerce")
        self.assertEqual(re.findall(r"^## .+$", text, re.MULTILINE), [
            "## How to run", "## Part 1 findings", "## Part 2 findings",
            "## Part 3 findings", "## Part 4 findings", "## Model-choice memo",
            "## Limitations",
        ])
        memo = text.split("## Model-choice memo\n", 1)[1].split("\n## Limitations", 1)[0]
        self.assertLessEqual(len(memo.split()), 300)
        self.assertIn("To: Head of the retention team", memo)
        self.assertIn("0.15", memo)
        for key in ("cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody"):
            self.assertIn(f"INR {self.results[2][key]}", memo)
        limitations = text.split("## Limitations\n", 1)[1]
        self.assertGreaterEqual(len(re.findall(r"^- ", limitations, re.MULTILINE)), 3)
        self.assertNotIn("$", text)
        url_lines = [line for line in text.splitlines() if re.search(r"https?://", line)]
        self.assertEqual(len(url_lines), 1)
        self.assertRegex(
            url_lines[0], r"^git clone https://github\.com/[^/\s]+/[^/\s]+\.git$"
        )
        run_section = text.split("## How to run\n", 1)[1].split("\n## Part 1 findings", 1)[0]
        commands = [
            "pip install -r requirements.txt", "python generate_data.py",
            "python part1.py", "python part2.py", "python part3.py", "python part4.py",
        ]
        positions = [run_section.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual((ROOT / "requirements.txt").read_text().strip(), f"numpy=={np.__version__}")

    def test_readme_uses_full_precision_wcss_drops_and_output_checkpoints(self):
        text = (ROOT / "README.md").read_text()
        X, _, _ = common.load_data()
        z, _, _ = part3.standardise(X[:, :5])
        wcss = {k: part3.kmeans(z, k)[2] for k in range(2, 7)}
        for k in range(2, 6):
            self.assertIn(f"{wcss[k] - wcss[k + 1]:.4f}", text)
        required_quoted_keys = {
            1: (
                "z_stat", "n_required", "gd_slope_gap", "resid_sd_ratio_raw",
                "resid_sd_ratio_log", "mr_adj_r2", "loglog_adj_r2",
            ),
            2: (
                "blind_accuracy", "tau_0.5_accuracy", "tau_0.5_recall",
                "tau_0.3_recall", "tau_0.5_precision", "tau_0.3_precision",
                "tau_0.3_f1", "knn_f1", "knn_recall", "cost_test_at_tau",
                "cost_test_at_0.5", "cost_test_flag_nobody",
            ),
            3: ("mean_pc1_segment_A", "mean_pc1_segment_B", "mean_pc1_segment_C"),
            4: ("round1_error", "round2_error", "round1_alpha", "round2_alpha", "rnn_total"),
        }
        for part, keys in required_quoted_keys.items():
            for key in keys:
                with self.subTest(part=part, key=key):
                    self.assertIn(self.results[part][key], text)


if __name__ == "__main__":
    unittest.main()
