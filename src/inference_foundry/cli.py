import argparse
import json
from pathlib import Path
from .experiment import experiment
from .policy import RoutingConstraints


def main():
    parser = argparse.ArgumentParser(description="Calibrate and measure a quality-constrained model router")
    parser.add_argument("command", choices=["benchmark", "serve"])
    parser.add_argument("--quality-floor", type=float, default=.94)
    parser.add_argument("--strong-budget", type=float, default=.4)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    constraints = RoutingConstraints(args.quality_floor, args.strong_budget)
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(constraints), host="127.0.0.1", port=args.port)
        return 0
    report = experiment(constraints)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if report["selection"]:
        paired = [{"example_id": item["example_id"], "variant": "baseline" if item["mode"] == "strong-only" else "candidate", "cohort": f"digit:{item['expected']}", "expected": item["expected"], "predicted": item["predicted"], "latency_ms": item["latency_ms"]} for item in report["observations"] if item["mode"] in ("strong-only", "routed")]
        (args.output / "releaseguard-observations.json").write_text(json.dumps(paired, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Inference Foundry — v0.1", "", f"Selection status: **{report['status']}**", "", "| Held-out policy | Accuracy | Strong-model calls | Mean compute ms | p95 compute ms |", "| --- | ---: | ---: | ---: | ---: |"]
    for mode, result in report["test"].items():
        lines.append(f"| {mode} | {result['accuracy']:.2%} | {result['strong_fraction']:.2%} | {result['latency_ms_mean']:.4f} | {result['latency_ms_p95']:.4f} |")
    lines.extend(["", f"Held-out constraints met: {report.get('test_constraints_met', 'No feasible policy')}", "", "## Interpretation limits", "", *[f"- {item}" for item in report["limitations"]], "", "See report.json for split/model hashes, calibration frontier, and per-example measured observations."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "test": report["test"], "test_constraints_met": report.get("test_constraints_met"), "report": str((args.output / 'report.md').resolve())}, indent=2))
    return 0 if report.get("test_constraints_met") else 3
