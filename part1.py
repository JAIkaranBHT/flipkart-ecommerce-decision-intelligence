"""Estimate order value and build transparent regression baselines.

Run with ``python part1.py`` from any working directory. All reported results
use the complete dataset; the value and X4 columns are in hundreds of INR.
"""

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np

from common import load_data, write_results


@dataclass
class LineFit:
    """The fitted line and the quantities needed to inspect its errors."""

    intercept: float
    slope: float
    fitted: np.ndarray
    residuals: np.ndarray
    tss: float
    ess: float
    rss: float
    r2: float
    rmse: float
    mae: float


def mean_and_sample_sd(values):
    """Compute the mean and sample SD, explicitly dividing variance by n - 1."""
    n = len(values)
    if n < 2:
        raise ValueError("A sample SD requires at least two observations.")
    mean = float(np.sum(values) / n)
    centered = values - mean
    sd = sqrt(float(np.sum(centered * centered)) / (n - 1))
    return mean, sd


def fit_simple_line(x, y):
    """Fit y = intercept + slope * x using sample covariance / sample variance."""
    n = len(y)
    mean_x, _ = mean_and_sample_sd(x)
    mean_y, _ = mean_and_sample_sd(y)
    dx = x - mean_x
    dy = y - mean_y
    sample_covariance = float(np.sum(dx * dy) / (n - 1))
    sample_variance_x = float(np.sum(dx * dx) / (n - 1))
    if sample_variance_x == 0:
        raise ValueError("A regression predictor must not be constant.")

    slope = sample_covariance / sample_variance_x
    intercept = mean_y - slope * mean_x
    fitted = intercept + slope * x
    residuals = y - fitted
    tss = float(np.sum(dy * dy))
    ess = float(np.sum((fitted - mean_y) ** 2))
    rss = float(np.sum(residuals * residuals))
    return LineFit(
        intercept=intercept,
        slope=slope,
        fitted=fitted,
        residuals=residuals,
        tss=tss,
        ess=ess,
        rss=rss,
        r2=1.0 - rss / tss,
        rmse=sqrt(rss / n),
        mae=float(np.sum(np.abs(residuals)) / n),
    )


def fit_line_by_gradient_descent(x, y):
    """Use 500 full-batch updates, recording costs after the specified updates."""
    n = len(y)
    mean_x, sd_x = mean_and_sample_sd(x)
    z = (x - mean_x) / sd_x
    slope_z = 0.0
    intercept_z = 0.0
    learning_rate = 0.1
    results = {"gd_J0": float(np.sum(y * y) / n)}

    for update in range(1, 501):
        residuals = y - (slope_z * z + intercept_z)
        # Both gradients use the same pre-update residuals.
        slope_gradient = float(-2.0 * np.sum(z * residuals) / n)
        intercept_gradient = float(-2.0 * np.sum(residuals) / n)
        slope_z -= learning_rate * slope_gradient
        intercept_z -= learning_rate * intercept_gradient

        if update in (1, 10, 100, 500):
            new_residuals = y - (slope_z * z + intercept_z)
            results[f"gd_J{update}"] = float(
                np.sum(new_residuals * new_residuals) / n
            )

    results["gd_slope"] = slope_z / sd_x
    results["gd_intercept"] = intercept_z - slope_z * mean_x / sd_x
    return results


def residual_spread_by_fitted_half(fit):
    """Keep file order for tied fitted values, then compare the two equal halves."""
    n = len(fit.fitted)
    if n % 2:
        raise ValueError("The pinned residual comparison requires equal halves.")
    order = np.argsort(fit.fitted, kind="stable")
    _, lower_sd = mean_and_sample_sd(fit.residuals[order[: n // 2]])
    _, upper_sd = mean_and_sample_sd(fit.residuals[order[n // 2 :]])
    return lower_sd, upper_sd, upper_sd / lower_sd


def adjusted_r_squared(r_squared, n, predictors):
    """Penalise the number of slopes, excluding the intercept from the count."""
    return 1.0 - (1.0 - r_squared) * (n - 1) / (n - predictors - 1)


def compute_results(features, value):
    """Compute Tasks 1.1 to 1.7 without rounding any intermediate quantity."""
    n = len(value)
    mean_value, sd_value = mean_and_sample_sd(value)
    se_value = sd_value / sqrt(n)
    critical_value = 1.96
    planning_mean = 8.0
    target_margin = 0.5
    z_stat = (mean_value - planning_mean) / se_value
    results = {
        "n": n,
        "mean_value": mean_value,
        "sd_value": sd_value,
        "se_value": se_value,
        "ci95_low": mean_value - critical_value * se_value,
        "ci95_high": mean_value + critical_value * se_value,
        "z_stat": z_stat,
        "reject_h0": abs(z_stat) > critical_value,
        "n_required": ceil((critical_value * sd_value / target_margin) ** 2),
    }

    # Task 1.4: one predictor on the original, hundreds-of-INR scale.
    x4 = features[:, 3]
    raw_fit = fit_simple_line(x4, value)
    results.update(
        {
            "reg_slope": raw_fit.slope,
            "reg_intercept": raw_fit.intercept,
            "tss": raw_fit.tss,
            "ess": raw_fit.ess,
            "rss": raw_fit.rss,
            "tss_equals_ess_plus_rss": (
                abs(raw_fit.tss - raw_fit.ess - raw_fit.rss)
                < 1e-6 * raw_fit.tss
            ),
            "r2": raw_fit.r2,
            "rmse": raw_fit.rmse,
            "mae": raw_fit.mae,
        }
    )

    # Task 1.5: independently estimate the same line by minimising mean SSE.
    gd_results = fit_line_by_gradient_descent(x4, value)
    results.update(gd_results)
    results["gd_slope_gap"] = abs(gd_results["gd_slope"] - raw_fit.slope)

    # Task 1.6: each model's halves are defined by its own fitted values.
    lower_raw, upper_raw, ratio_raw = residual_spread_by_fitted_half(raw_fit)
    if np.any(value <= 0) or np.any(x4 <= 0):
        raise ValueError("The log models require strictly positive value and X4.")
    log_value = np.log(value)
    log_x4 = np.log(x4)
    log_fit = fit_simple_line(log_x4, log_value)
    lower_log, upper_log, ratio_log = residual_spread_by_fitted_half(log_fit)
    results.update(
        {
            "resid_sd_lower_raw": lower_raw,
            "resid_sd_upper_raw": upper_raw,
            "resid_sd_ratio_raw": ratio_raw,
            "loglog_intercept": log_fit.intercept,
            "loglog_slope": log_fit.slope,
            "loglog_r2": log_fit.r2,
            "loglog_rmse": log_fit.rmse,
            "resid_sd_lower_log": lower_log,
            "resid_sd_upper_log": upper_log,
            "resid_sd_ratio_log": ratio_log,
        }
    )

    # Task 1.7: [constant, ln(X4), X1, X2, X3, X5, X6, X7], in that order.
    design = np.column_stack(
        (
            np.ones(n),
            log_x4,
            features[:, 0],
            features[:, 1],
            features[:, 2],
            features[:, 4],
            features[:, 5],
            features[:, 6],
        )
    )
    normal_matrix = design.T @ design
    normal_target = design.T @ log_value
    beta = np.linalg.solve(normal_matrix, normal_target)
    multiple_residuals = log_value - design @ beta
    multiple_rss = float(np.sum(multiple_residuals * multiple_residuals))
    multiple_r2 = 1.0 - multiple_rss / log_fit.tss
    for index, coefficient in enumerate(beta):
        results[f"mr_beta_{index}"] = float(coefficient)
    results["mr_r2"] = multiple_r2
    results["mr_adj_r2"] = adjusted_r_squared(multiple_r2, n, 7)
    results["loglog_adj_r2"] = adjusted_r_squared(log_fit.r2, n, 1)
    return results


def main():
    features, _, value = load_data()
    write_results("part1_results.txt", compute_results(features, value))
    print("Part 1 complete: outputs/part1_results.txt")


if __name__ == "__main__":
    main()
