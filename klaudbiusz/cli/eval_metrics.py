"""Shared evaluation metric calculations for Klaudbiusz evaluation framework."""

from math import prod
from typing import Protocol


class MetricsProtocol(Protocol):
    """Protocol for metrics objects that can be scored."""

    build_success: bool
    runtime_success: bool
    type_safety: bool
    tests_pass: bool
    databricks_connectivity: bool
    local_runability_score: int
    deployability_score: int

    # Different between evaluate_app.py and evaluate_all.py
    # evaluate_app uses: data_validity_score (int 0-5), ui_functional_score (int 0-5)
    # evaluate_all uses: data_returned (bool), ui_renders (bool)


def _gm(values):
    """Geometric mean of values, with floor at 1e-6."""
    vals = [max(v, 1e-6) for v in values if v is not None]
    return prod(vals) ** (1.0 / len(vals)) if vals else 0.0


def _to01_bool(x):
    """Convert boolean to 0/1 float."""
    return 1.0 if x else 0.0


def _to01_5(x):
    """Convert 0-5 score to 0-1 range."""
    return max(0.0, min(1.0, (x or 0) / 5.0))


def calculate_appeval_100(
    build_success: bool,
    runtime_success: bool,
    type_safety: bool,
    tests_pass: bool,
    databricks_connectivity: bool,
    data_metric: float | bool,  # Either 0-5 score or boolean
    ui_metric: float | bool,  # Either 0-5 score or boolean
    local_runability_score: int,
    deployability_score: int,
) -> float:
    """
    Calculate the appeval_100 composite score.

    Formula:
        R = geometric_mean([build, runtime, type_safety, tests, db_connectivity, data, ui])
        D = geometric_mean([local_runability_score/5, deployability_score/5])
        G = (0.25 + 0.75*build) * (0.25 + 0.75*runtime) * (0.50 + 0.50*db_connectivity)
        appeval_100 = 100 * (0.7*R + 0.3*D) * G

    Args:
        build_success: Binary metric
        runtime_success: Binary metric
        type_safety: Binary metric
        tests_pass: Binary metric
        databricks_connectivity: Binary metric
        data_metric: Either a 0-5 score (evaluate_app) or boolean (evaluate_all)
        ui_metric: Either a 0-5 score (evaluate_app) or boolean (evaluate_all)
        local_runability_score: DevX score 0-5
        deployability_score: DevX score 0-5

    Returns:
        Score from 0-100
    """
    # Normalize data and UI metrics
    if isinstance(data_metric, bool):
        data_normalized = _to01_bool(data_metric)
    else:
        data_normalized = _to01_5(data_metric)

    if isinstance(ui_metric, bool):
        ui_normalized = _to01_bool(ui_metric)
    else:
        ui_normalized = _to01_5(ui_metric)

    # Calculate R (reliability/functionality)
    R = _gm([
        _to01_bool(build_success),
        _to01_bool(runtime_success),
        _to01_bool(type_safety),
        _to01_bool(tests_pass),
        _to01_bool(databricks_connectivity),
        data_normalized,
        ui_normalized,
    ])

    # Calculate D (developer experience)
    D = _gm([
        _to01_5(local_runability_score),
        _to01_5(deployability_score),
    ])

    # Calculate G (gating factor)
    G = (0.25 + 0.75 * _to01_bool(build_success)) \
      * (0.25 + 0.75 * _to01_bool(runtime_success)) \
      * (0.50 + 0.50 * _to01_bool(databricks_connectivity))

    # Final score
    appeval_100 = 100.0 * (0.7 * R + 0.3 * D) * G
    return round(appeval_100, 1)
