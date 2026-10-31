import time
from threading import BoundedSemaphore
from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .experiment import calibrate, predict_one
from .models import train_bundle
from .policy import RoutingConstraints


class PredictionRequest(BaseModel):
    pixels: Annotated[list[Annotated[float, Field(ge=0, le=16, allow_inf_nan=False)]], Field(min_length=64, max_length=64)]


def create_app(constraints: RoutingConstraints = RoutingConstraints()):
    bundle = train_bundle()
    chosen, _ = calibrate(bundle, constraints)
    app = FastAPI(title="Inference Foundry", version="0.1")
    capacity = BoundedSemaphore(8)

    @app.get("/health")
    def health():
        return {"status": "ready" if chosen else "infeasible", "policy": chosen, "model_bundle_sha256": bundle["provenance"]["model_bundle_sha256"]}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if chosen is None:
            raise HTTPException(503, "No calibration policy satisfies the configured constraints")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Inference capacity reached", headers={"Retry-After": "1"})
        try:
            start = time.perf_counter_ns()
            result = predict_one(bundle, request.pixels, chosen["threshold"])
            result["compute_ms"] = (time.perf_counter_ns() - start) / 1_000_000
            result["threshold"] = chosen["threshold"]
            return result
        finally:
            capacity.release()

    return app
