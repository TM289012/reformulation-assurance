"""Process-window experiment design for Reformulation Assurance v0.6.2.

The designer preserves the confirmed formulation and deliberately varies selected
process controls around their nominal settings. It supports compact one-factor
studies, corner-plus-center studies, and full three-level grids.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence
import hashlib
import json

import pandas as pd


DESIGN_MODES = {
    "One factor at a time": "ofat",
    "Corners plus center": "corners",
    "Full three-level grid": "grid3",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def design_fingerprint(
    *,
    source_replicate_group: str,
    process_columns: Sequence[str],
    deltas: Mapping[str, float],
    mode: str,
    nominal_inputs: Mapping[str, Any],
) -> str:
    payload = {
        "source_replicate_group": source_replicate_group,
        "process_columns": list(process_columns),
        "deltas": {key: float(deltas[key]) for key in sorted(deltas)},
        "mode": mode,
        "nominal_inputs": dict(nominal_inputs),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _level_vectors(count: int, mode: str) -> list[tuple[int, ...]]:
    if count < 1:
        raise ValueError("select at least one process variable")
    if mode == "ofat":
        vectors: list[tuple[int, ...]] = [(0,) * count]
        for index in range(count):
            low = [0] * count
            high = [0] * count
            low[index] = -1
            high[index] = 1
            vectors.extend([tuple(low), tuple(high)])
        return vectors
    if mode == "corners":
        return [(0,) * count, *list(product((-1, 1), repeat=count))]
    if mode == "grid3":
        if count > 3:
            raise ValueError("the full three-level grid is limited to three process variables")
        return list(product((-1, 0, 1), repeat=count))
    raise ValueError(f"unsupported process-window design mode: {mode}")


def design_process_window(
    *,
    config: Mapping[str, Any],
    nominal_inputs: Mapping[str, Any],
    source_replicate_group: str,
    process_columns: Sequence[str],
    deltas: Mapping[str, float],
    mode: str = "corners",
) -> pd.DataFrame:
    """Build a bounded process-window matrix around a confirmed candidate."""
    selected = list(dict.fromkeys(process_columns))
    available = list(config.get("process_columns", []))
    unknown = [column for column in selected if column not in available]
    if unknown:
        raise ValueError(f"unknown process variable(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("select at least one process variable")

    feature_columns = [
        *config.get("mixture_columns", []),
        *available,
        *config.get("categorical_columns", []),
    ]
    missing = [column for column in feature_columns if column not in nominal_inputs]
    if missing:
        raise ValueError(f"candidate is missing required input(s): {', '.join(missing)}")

    for column in selected:
        delta = float(deltas.get(column, 0.0))
        if delta <= 0:
            raise ValueError(f"process-window delta for {column} must be greater than zero")

    fingerprint = design_fingerprint(
        source_replicate_group=source_replicate_group,
        process_columns=selected,
        deltas=deltas,
        mode=mode,
        nominal_inputs={column: nominal_inputs[column] for column in feature_columns},
    )

    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for levels in _level_vectors(len(selected), mode):
        row = {column: nominal_inputs[column] for column in feature_columns}
        labels: list[str] = []
        for column, level in zip(selected, levels):
            nominal = float(nominal_inputs[column])
            lo, hi = map(float, config["process_bounds"][column])
            value = min(max(nominal + level * float(deltas[column]), lo), hi)
            row[column] = value
            labels.append(f"{column}={'low' if level < 0 else 'high' if level > 0 else 'nominal'}")
        signature = tuple(row[column] for column in feature_columns)
        if signature in seen:
            continue
        seen.add(signature)
        row.update(
            {
                "purpose": "Process window: " + ", ".join(labels),
                "process_window_design": mode,
                "process_window_source_group": source_replicate_group,
                "process_window_levels": {column: int(level) for column, level in zip(selected, levels)},
                "process_window_deltas": {column: float(deltas[column]) for column in selected},
                "design_fingerprint": fingerprint,
            }
        )
        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("process-window design produced no unique experiments")
    return result
