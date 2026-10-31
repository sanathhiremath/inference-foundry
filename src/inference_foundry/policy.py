from dataclasses import dataclass
from math import isfinite
import numpy as np


@dataclass(frozen=True)
class RoutingConstraints:
    quality_floor: float = 0.94
    max_strong_fraction: float = 0.40

    def __post_init__(self):
        if any(not isfinite(value) or not 0 <= value <= 1 for value in (self.quality_floor, self.max_strong_fraction)):
            raise ValueError("Constraints must be finite fractions in [0, 1]")


def select_threshold(confidence, cheap_labels, strong_labels, expected, constraints: RoutingConstraints):
    confidence = np.asarray(confidence, dtype=float)
    cheap_labels = np.asarray(cheap_labels)
    strong_labels = np.asarray(strong_labels)
    expected = np.asarray(expected)
    if confidence.ndim != 1 or not len(confidence) or any(values.shape != confidence.shape for values in (cheap_labels, strong_labels, expected)):
        raise ValueError("Calibration arrays must be aligned nonempty vectors")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must be finite and in [0, 1]")
    # Evaluate observed confidence cut points, not test-set performance.
    thresholds = sorted({0.0, 1.0, *[float(np.nextafter(value, np.inf)) for value in confidence]})
    frontier = []
    for threshold in thresholds:
        use_strong = confidence < threshold
        predictions = np.where(use_strong, strong_labels, cheap_labels)
        accuracy = float(np.mean(predictions == expected))
        fraction = float(np.mean(use_strong))
        frontier.append({"threshold": threshold, "accuracy": accuracy, "strong_fraction": fraction, "feasible": accuracy >= constraints.quality_floor and fraction <= constraints.max_strong_fraction})
    feasible = [point for point in frontier if point["feasible"]]
    chosen = min(feasible, key=lambda point: (point["strong_fraction"], -point["accuracy"], point["threshold"])) if feasible else None
    return chosen, frontier
