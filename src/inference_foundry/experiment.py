import math
import time
import numpy as np
from .models import pooled, train_bundle
from .policy import RoutingConstraints, select_threshold


def calibrate(bundle, constraints):
    ids = bundle["calibration"]
    probabilities = bundle["cheap"].predict_proba(pooled(bundle["data"][ids]))
    cheap_labels = bundle["cheap"].classes_[probabilities.argmax(axis=1)]
    return select_threshold(probabilities.max(axis=1), cheap_labels, bundle["strong"].predict(bundle["data"][ids]), bundle["labels"][ids], constraints)


def predict_one(bundle, features, threshold):
    values = np.asarray(features, dtype=float)
    if values.shape != (64,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 16)):
        raise ValueError("Expected 64 finite pixels in [0, 16]")
    probabilities = bundle["cheap"].predict_proba(pooled(values.reshape(1, 64)))[0]
    confidence = float(probabilities.max())
    use_strong = confidence < threshold
    label = int(bundle["strong"].predict(values.reshape(1, 64))[0]) if use_strong else int(bundle["cheap"].classes_[probabilities.argmax()])
    return {"prediction": label, "path": "strong" if use_strong else "cheap", "cheap_confidence": confidence}


def experiment(constraints: RoutingConstraints) -> dict:
    bundle = train_bundle()
    chosen, frontier = calibrate(bundle, constraints)
    report = {"version": "0.1", "constraints": {"quality_floor": constraints.quality_floor, "max_strong_fraction": constraints.max_strong_fraction}, "status": "CALIBRATED" if chosen else "INFEASIBLE", "selection": chosen, "calibration_frontier": frontier, "provenance": bundle["provenance"], "test": {}, "observations": [], "limitations": ["Bundled digits is a small reference workload, not a production task.", "A calibrated invocation budget is not a hard runtime cost cap.", "Maximum class probability is an uncalibrated confidence score, not a correctness probability.", "Single-process wall-clock latency includes routing overhead but excludes HTTP/network/concurrent load.", "The held-out test is evaluated once after policy selection. Do not tune thresholds against its results."]}
    modes = ["cheap-only", "strong-only"] + (["routed"] if chosen else [])
    for offset, index in enumerate(bundle["test"]):
        # Rotate run order instead of systematically giving one policy the warm cache.
        order = modes[offset % len(modes):] + modes[:offset % len(modes)]
        for mode in order:
            values = bundle["data"][index:index+1]
            start = time.perf_counter_ns()
            if mode == "cheap-only":
                label = int(bundle["cheap"].predict(pooled(values))[0])
                path = "cheap"
            elif mode == "strong-only":
                label = int(bundle["strong"].predict(values)[0])
                path = "strong"
            else:
                decision = predict_one(bundle, values[0], chosen["threshold"])
                label, path = decision["prediction"], decision["path"]
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            report["observations"].append({"example_id": f"digits:{int(index)}", "mode": mode, "path": path, "expected": int(bundle["labels"][index]), "predicted": label, "latency_ms": elapsed})
    for mode in modes:
        items = [item for item in report["observations"] if item["mode"] == mode]
        timings = sorted(item["latency_ms"] for item in items)
        report["test"][mode] = {"count": len(items), "accuracy": sum(item["predicted"] == item["expected"] for item in items) / len(items), "strong_fraction": sum(item["path"] == "strong" for item in items) / len(items), "latency_ms_mean": sum(timings) / len(timings), "latency_ms_p50": timings[len(timings)//2], "latency_ms_p95": timings[math.ceil(len(timings)*.95)-1]}
    if chosen:
        routed = report["test"]["routed"]
        report["test_constraints_met"] = routed["accuracy"] >= constraints.quality_floor and routed["strong_fraction"] <= constraints.max_strong_fraction
    return report
