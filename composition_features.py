"""Boundary-aware feature representations for mixture data.

The default response models work on raw mass fractions. That treats an
ingredient at 0 percent as merely a very low dose, although "absent" is often a
different regime from "present at a trace" (an emulsifier that is or is not in
the system, a pigment that is or is not there). Two reviewers on the Materials
Science Community Discourse (November 2026) suggested testing a boundary-aware
representation as an ablation against the current kernel rather than as a swap,
and this module is that representation. Nothing here is used unless asked for:
``MixedFeatureEncoder`` keeps raw fractions unless it is given a
``CompositionSpec`` with a representation other than ``"raw"``.

Two representations are offered, both derived from the raw columns alone (no new
information, a different encoding of the same rows):

``presence``
    Adds a presence indicator ``I(x_i > 0)`` for every mixture ingredient that is
    absent from at least one and present in at least one of the fitted lots. The
    raw fraction stays as the dosage, so a kernel with per-variable length scales
    can put a short scale on the indicator (a jump at the boundary) and a long one
    on the dose.

``presence_ilr``
    The indicators above, plus the isometric log-ratio (ilr) coordinates of the
    whole mixture composition in place of its raw fractions. Zeros are handled by
    multiplicative replacement (Martín-Fernández, Barceló-Vidal and
    Pawlowsky-Glahn, 2003): each zero becomes a small value delta, 65 percent of the
    smallest non-zero amount of that ingredient in the fitted lots, and the other
    parts of the row shrink proportionally so the total is unchanged. The ilr basis
    comes from a sequential binary partition that follows the problem: first the
    reformulation slot (the ingredient being removed and the ingredients that may
    be absent, its substitutes) against the rest of the formula, then the removed
    ingredient against its substitutes, then within each group in column order.
    Process variables and categorical variables are unchanged.

Egozcue, Pawlowsky-Glahn, Mateu-Figueras and Barceló-Vidal (2003), "Isometric
logratio transformations for compositional data analysis", Mathematical Geology
35(3). Martín-Fernández, Barceló-Vidal and Pawlowsky-Glahn (2003), "Dealing with
zeros and missing values in compositional data sets using nonparametric
imputation", Mathematical Geology 35(3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPRESENTATIONS = ("raw", "presence", "presence_ilr")


@dataclass(frozen=True)
class CompositionSpec:
    """Which columns form the mixture and how to represent them.

    ``mixture_columns`` are all the ingredient columns, which sum to ``total`` in
    every row. ``removed`` names the ingredient being reformulated out (it may be
    None). ``optional`` lists the ingredients that may be absent, normally the
    removed ingredient and its substitutes; when it is empty the ingredients that
    have a zero in the fitted data are used. ``representation`` is one of
    ``REPRESENTATIONS``.
    """

    mixture_columns: tuple[str, ...]
    removed: str | None = None
    optional: tuple[str, ...] = field(default_factory=tuple)
    total: float = 100.0
    representation: str = "raw"
    zero_tolerance: float = 1e-9
    replacement_fraction: float = 0.65

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"unknown representation '{self.representation}'; "
                f"choose one of {', '.join(REPRESENTATIONS)}"
            )
        if len(self.mixture_columns) < 2 and self.representation == "presence_ilr":
            raise ValueError("ilr coordinates need at least two mixture columns")
        if self.removed is not None and self.removed not in self.mixture_columns:
            raise ValueError(f"removed ingredient '{self.removed}' is not a mixture column")
        for column in self.optional:
            if column not in self.mixture_columns:
                raise ValueError(f"optional ingredient '{column}' is not a mixture column")
        if not 0.0 < self.replacement_fraction < 1.0:
            raise ValueError("replacement_fraction must lie strictly between 0 and 1")
        if self.total <= 0:
            raise ValueError("mixture total must be positive")

    @property
    def uses_ilr(self) -> bool:
        return self.representation == "presence_ilr"

    @property
    def uses_indicators(self) -> bool:
        return self.representation in ("presence", "presence_ilr")


def composition_spec_from_config(config: Mapping[str, object], representation: str) -> CompositionSpec:
    """Build a spec from a project configuration dictionary.

    The optional ingredients are the removed ingredient plus every mixture column
    whose lower bound is zero, which is how a project says an ingredient may be
    left out.
    """
    mixture_columns = tuple(str(c) for c in config["mixture_columns"])
    removed = config.get("ingredient_to_remove")
    removed = str(removed) if removed in mixture_columns else None
    bounds = config.get("mixture_bounds") or {}
    optional = []
    for column in mixture_columns:
        lower = None
        if isinstance(bounds, Mapping) and column in bounds:
            lower = float(bounds[column][0])
        if column == removed or (lower is not None and lower <= 0.0):
            optional.append(column)
    return CompositionSpec(
        mixture_columns=mixture_columns,
        removed=removed,
        optional=tuple(optional),
        total=float(config.get("mixture_total", 100.0)),
        representation=representation,
    )


def sequential_binary_partition(
    mixture_columns: Sequence[str],
    slot: Sequence[str],
    removed: str | None,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Ordered balances (numerator group, denominator group), D - 1 of them.

    The first balance sets the reformulation slot against the rest of the formula,
    the second the removed ingredient against its substitutes, and the remaining
    balances split each group sequentially in column order. Every mixture column
    appears in the tree exactly once, so the balances are an orthonormal ilr basis.
    """
    columns = list(mixture_columns)
    slot_columns = [c for c in columns if c in set(slot)]
    body_columns = [c for c in columns if c not in set(slot)]

    def sequential(group: list[str]):
        if len(group) <= 1:
            return group[0] if group else None
        return (group[0], sequential(group[1:]))

    removed_part = [c for c in slot_columns if c == removed]
    substitutes = [c for c in slot_columns if c != removed]
    if removed_part and substitutes:
        slot_tree = (sequential(removed_part), sequential(substitutes))
    else:
        slot_tree = sequential(slot_columns)
    body_tree = sequential(body_columns)
    if slot_tree is None:
        tree = body_tree
    elif body_tree is None:
        tree = slot_tree
    else:
        tree = (slot_tree, body_tree)

    def leaves(node) -> tuple[str, ...]:
        if isinstance(node, tuple):
            return leaves(node[0]) + leaves(node[1])
        return (node,)

    balances: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def walk(node) -> None:
        if isinstance(node, tuple):
            balances.append((leaves(node[0]), leaves(node[1])))
            walk(node[0])
            walk(node[1])

    walk(tree)
    return balances


def multiplicative_replacement(
    values: np.ndarray, deltas: np.ndarray, total: float, zero_tolerance: float = 1e-9
) -> np.ndarray:
    """Replace zeros by ``deltas`` and shrink the non-zero parts so rows keep their total."""
    X = np.asarray(values, dtype=float).copy()
    deltas = np.asarray(deltas, dtype=float)
    zero = X <= zero_tolerance
    for row in range(X.shape[0]):
        mask = zero[row]
        if not mask.any():
            continue
        added = float(deltas[mask].sum())
        X[row, ~mask] *= 1.0 - added / total
        X[row, mask] = deltas[mask]
    return X


def ilr_balances(
    values: np.ndarray,
    mixture_columns: Sequence[str],
    partition: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> np.ndarray:
    """Isometric log-ratio coordinates of strictly positive rows for the given partition."""
    X = np.asarray(values, dtype=float)
    if np.any(X <= 0):
        raise ValueError("ilr coordinates need strictly positive parts; replace zeros first")
    index = {column: i for i, column in enumerate(mixture_columns)}
    logs = np.log(X)
    out = np.empty((X.shape[0], len(partition)), dtype=float)
    for k, (numerator, denominator) in enumerate(partition):
        r, s = len(numerator), len(denominator)
        num = logs[:, [index[c] for c in numerator]].mean(axis=1)
        den = logs[:, [index[c] for c in denominator]].mean(axis=1)
        out[:, k] = np.sqrt(r * s / (r + s)) * (num - den)
    return out


class CompositionFeatures:
    """Fit-then-transform producer of the derived mixture features.

    ``fit`` decides which indicators are informative (an ingredient that is always
    present or always absent in the fitted rows gets none) and learns the zero
    replacement deltas; ``transform`` returns a numeric frame whose columns are
    listed in ``feature_names`` after fitting.
    """

    def __init__(self, spec: CompositionSpec):
        self.spec = spec
        self.indicator_columns: list[str] = []
        self.deltas: np.ndarray | None = None
        self.partition: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self._fitted = False

    def _matrix(self, data: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.spec.mixture_columns if c not in data.columns]
        if missing:
            raise KeyError(f"composition columns missing from the data: {', '.join(missing)}")
        X = data[list(self.spec.mixture_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError("composition columns contain missing or non-finite values")
        if np.any(X < -self.spec.zero_tolerance):
            raise ValueError("mixture fractions cannot be negative")
        return np.clip(X, 0.0, None)

    def fit(self, data: pd.DataFrame) -> "CompositionFeatures":
        X = self._matrix(data)
        tol = self.spec.zero_tolerance
        present = X > tol
        candidates = list(self.spec.optional) or list(self.spec.mixture_columns)
        self.indicator_columns = []
        if self.spec.uses_indicators:
            for i, column in enumerate(self.spec.mixture_columns):
                if column in candidates and present[:, i].any() and (~present[:, i]).any():
                    self.indicator_columns.append(column)
        if self.spec.uses_ilr:
            deltas = np.empty(X.shape[1], dtype=float)
            for i in range(X.shape[1]):
                nonzero = X[present[:, i], i]
                smallest = float(nonzero.min()) if nonzero.size else self.spec.total * 1e-3
                deltas[i] = self.spec.replacement_fraction * smallest
            self.deltas = deltas
            slot = list(self.spec.optional) or [
                c for i, c in enumerate(self.spec.mixture_columns) if (~present[:, i]).any()
            ]
            self.partition = sequential_binary_partition(self.spec.mixture_columns, slot, self.spec.removed)
        self._fitted = True
        return self

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("composition features must be fitted first")
        names = [f"present_{column}" for column in self.indicator_columns]
        if self.spec.uses_ilr:
            names.extend(f"ilr_{k + 1}" for k in range(len(self.partition)))
        return names

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("composition features must be fitted first")
        X = self._matrix(data)
        columns: dict[str, np.ndarray] = {}
        index = {column: i for i, column in enumerate(self.spec.mixture_columns)}
        for column in self.indicator_columns:
            columns[f"present_{column}"] = (X[:, index[column]] > self.spec.zero_tolerance).astype(float)
        if self.spec.uses_ilr:
            assert self.deltas is not None
            replaced = multiplicative_replacement(X, self.deltas, self.spec.total, self.spec.zero_tolerance)
            balances = ilr_balances(replaced, self.spec.mixture_columns, self.partition)
            for k in range(balances.shape[1]):
                columns[f"ilr_{k + 1}"] = balances[:, k]
        return pd.DataFrame(columns, index=data.index)

    def describe(self) -> str:
        """One line for a model note or a log."""
        if not self._fitted:
            return f"composition features ({self.spec.representation}), not fitted"
        parts = [f"representation={self.spec.representation}"]
        parts.append(
            "indicators=" + (", ".join(self.indicator_columns) if self.indicator_columns else "none")
        )
        if self.spec.uses_ilr:
            parts.append(f"ilr coordinates={len(self.partition)}")
        return "; ".join(parts)
