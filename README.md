# Inference Foundry

Version 0.1. Internal package: `inference_foundry`.

**When is a stronger model worth the extra compute?**

Inference Foundry is a working two-model routing prototype. It chooses an uncertainty threshold using a calibration split, under a quality floor and a strong-model invocation budget. It then evaluates that fixed policy on an untouched test split and compares it with cheap-only and strong-only baselines.

The product problem is conditional computation under constraints, not another digit classifier. Bundled handwritten digits provide a small, offline reference workload with real trained models and measured inference. No API key or dataset download is required after installing dependencies.

## Run

Python 3.11+, Windows or Ubuntu. Create a virtual environment and install the project with dev dependencies:

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\python -m pip install -e ".[dev]"`. Ubuntu: `.venv/bin/python -m pip install -e '.[dev]'`. Use that environment's Python for the commands below. With dependencies already installed, the convenience entry point works directly:

```bash
python run.py benchmark --quality-floor 0.94 --strong-budget 0.40
python run.py serve --port 8011
python -m pytest -q
```

`artifacts/report.md` contains the readable comparison. `artifacts/report.json` contains the calibration frontier, split/model hashes, and every test prediction and measured compute time. An infeasible calibration policy or a held-out constraint miss returns exit code 3. No threshold is silently relaxed to manufacture a passing result.

Interactive API docs: `http://127.0.0.1:8011/docs`.

## Local reference result

Measured on this Windows machine using Python 3.11.9 and scikit-learn 1.5.2; 360 held-out test examples:

| Policy | Accuracy | Strong-model calls | Mean compute ms | p95 compute ms |
| --- | ---: | ---: | ---: | ---: |
| Cheap only | 90.00% | 0% | 0.3433 | 1.0384 |
| Strong only | 98.06% | 100% | 4.0885 | 11.9103 |
| Calibrated router | 94.44% | 12.22% | 1.0509 | 3.4598 |

The router met the configured 94% observed-quality target using fewer strong-model invocations. It did **not** preserve the stronger model's 98.06% quality. This is an explicit quality/compute tradeoff. Timings vary by machine and run; no production throughput or dollar-cost claim is made.

## Decision flow

```text
Train split ──────> cheap model + strong model
                           |
Calibration split -> confidence cut points -> feasible policy selection
                                                   |
Untouched test split -> cheap-only / strong-only / fixed router
                                                   |
                          quality + invocation share + measured time
```

- **Cheap path:** 2×2 average pooling, 16 features, scaler, logistic regression.
- **Strong path:** 80-tree ExtraTrees classifier over all 64 input features.
- **Selection objective:** minimize the strong-model invocation fraction among policies meeting both calibration constraints.
- **No test-driven tuning:** the selection code only receives calibration data. A test changes all held-out labels and verifies the selected policy stays unchanged.
- **Tie handling:** examples with equal confidence move together; the policy cannot choose an arbitrary favorable subset.
- **Infeasibility:** no qualifying policy means no routed prediction service; `/predict` returns 503.
- **Bounded serving:** at most eight concurrent prediction handlers per process; overflow receives 429. API input rejects incorrect shapes, out-of-range pixels, and nonfinite values.

## API

`POST /predict` accepts `{"pixels": [64 numbers in the range 0..16]}`. It returns the predicted label, cheap-model confidence score, chosen path, threshold, and measured compute time. `GET /health` reports readiness and the fitted model-bundle fingerprint.

The service binds to loopback. It is a local prototype, not a public multi-tenant inference service. It does not claim distributed scheduling, authentication, online budget enforcement, or GPU batching.

## What the numbers do and do not establish

The invocation budget is calibrated offline; distribution shift can change its runtime rate. Classifier maximum probability is used as a routing score, not a calibrated probability of correctness. The quality floor is an observed constraint on this sample, not a statistical guarantee. Test quality should not be used to retune the same held-out test. Latency is per-example single-process wall time after warmup, excludes HTTP/network load, and includes routing overhead. Invocation share is a compute proxy, not a monetary cost model. The bundled workload is small and does not establish behavior on language or production vision tasks.

Next meaningful milestones: adapter contracts for a real application model, load and queueing experiments, batch-aware routing, drift monitoring, and quality confidence bounds.

## Can this router actually pass a release review?

The benchmark also exports `artifacts/releaseguard-observations.json`: paired **strong-only baseline** and **routed candidate** predictions from the real held-out evaluation. No outcomes are injected into that export.

From the workspace root, after running the benchmark:

```bash
python projects/releaseguard/run.py evaluate --input projects/inference-foundry/artifacts/releaseguard-observations.json --policy projects/inference-foundry/release-policy.json --output projects/inference-foundry/artifacts/release-review --enforce
```

The example review policy permits at most a five-percentage-point overall loss against strong-only, but retains an eight-point per-cohort limit and requires all ten cohorts. Minimum sample thresholds match the 360-example local test size. These are explicit demonstration choices, not production recommendations. A router can meet its global target and still be blocked by the cohort gate; do not relax the gate after observing a failure. Exit code 3 means blocked or insufficient evidence.

In the saved local review, ReleaseGuard blocked the routed candidate because `digit:1` and `digit:2` exceeded the per-cohort drop limit. Unlike the diagnostic scenarios inside ReleaseGuard's bundled demo, these were the models' actual held-out predictions, with no output injection. The router remains a blocked candidate under that review policy.

Reference: [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).

## Repository

Source: https://github.com/sanathhiremath/inference-foundry


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
