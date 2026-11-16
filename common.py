"""Data loading and text output shared by the four independent entry points."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FEATURE_NAMES = (
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
)


def load_data():
    """Return X1..X7, the binary outcome and the value, keeping file order."""
    with (ROOT / "data" / "records.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(header) != 10 or len(rows) != 800:
        raise ValueError("Expected the unmodified 800-row, 10-column dataset.")
    numbers = np.asarray([row[1:] for row in rows], dtype=float)
    return numbers[:, :7], numbers[:, 7].astype(int), numbers[:, 8]


def format_value(value):
    """Only round at the output boundary; keep counts and flags unambiguous."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def write_results(filename, results):
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / filename).open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in results.items():
            handle.write(f"{key} = {format_value(value)}\n")


def write_csv(filename, header, rows):
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(value) for value in row])
