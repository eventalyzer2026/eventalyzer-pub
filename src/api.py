from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import os
from typing import Optional
import time

import app.src.state as ass
from app.src.schemas import (
    ECSLog,
    OnlinePredictRequest,
    OnlinePredictResponse,
    OfflineTrainRequest,
    OfflineTrainResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchClassifyItemResponse,
)

from app.src.classificator import LogClassificatorModel
from app.src.clusterer import LogClustererModel
from app.src.vertorized import LogVertorizer

logging.basicConfig(level=logging.INFO)
MODEL_VERSION = os.getenv("MODEL_VERSION", "current")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading models...")

    logging.info(f"CWD: {os.getcwd()}")
    logging.info(f"Files in CWD: {os.listdir()}")

    ass.vectorizer = LogVertorizer(
        ohe_encoding_features=None,
        le_encoding_features=None,
        insufficent_columns=None,
        ohe_filepath="app/src/models/ohe_enc.joblib",
        le_filepath="app/src/models/le_enc.joblib"
    )

    ass.classifier = LogClassificatorModel("app/src/models/log_classifier_2.joblib")
    ass.clusterer = LogClustererModel("app/src/models/clusterer_2.joblib")

    yield

    logging.info("Shutting down application...")


app = FastAPI(title="Log ML Service", lifespan=lifespan)


def _predict_internal(log: ECSLog, return_proba: bool = False, model_version: Optional[str] = None) -> OnlinePredictResponse:
    if ass.vectorizer is None or ass.classifier is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    t1 = time.time()

    try:
        df = ass.vectorizer.ecs2pandas(log.model_dump_json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}")

    pred = ass.classifier.classify(df)

    proba = None
    detail = None

    if return_proba:
        proba_arr = ass.classifier.predict_proba(df)
        classes = getattr(ass.classifier.pipe, "classes_", None)
        if classes is not None:
            try:
                class_idx = list(classes).index(pred[0])
                proba = float(proba_arr[0][class_idx])
            except ValueError:
                proba = float(proba_arr.max())
        else:
            proba = float(proba_arr.max())

    if int(pred[0]) == -1:
        detail = "Event is anomaly"

    t2 = time.time()

    return OnlinePredictResponse(
        cluster=int(pred[0]),
        proba=proba,
        model_version=model_version or MODEL_VERSION,
        elapsed=t2 - t1,
        detail=detail,
    )


@app.post("/online/predict", response_model=OnlinePredictResponse)
def online_predict(request: OnlinePredictRequest):
    return _predict_internal(
        log=request.log,
        return_proba=request.return_proba,
        model_version=request.model_version,
    )


@app.post("/classify", response_model=OnlinePredictResponse)
def classify(log: ECSLog):
    return _predict_internal(log=log, return_proba=True)


@app.post("/batch/classify", response_model=BatchClassifyResponse)
def batch_classify(request: BatchClassifyRequest):
    if ass.vectorizer is None or ass.classifier is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    results: list[BatchClassifyItemResponse] = []
    succeeded = 0
    failed = 0

    for idx, log in enumerate(request.logs):
        try:
            response = _predict_internal(
                log=log,
                return_proba=request.return_proba,
                model_version=request.model_version,
            )
            results.append(
                BatchClassifyItemResponse(
                    index=idx,
                    cluster=response.cluster,
                    proba=response.proba,
                    model_version=response.model_version,
                    detail=response.detail,
                    elapsed=response.elapsed,
                )
            )
            succeeded += 1
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            results.append(
                BatchClassifyItemResponse(
                    index=idx,
                    model_version=request.model_version or MODEL_VERSION,
                    error=f"HTTP {exc.status_code}: {detail}",
                )
            )
            failed += 1
        except Exception as exc:
            results.append(
                BatchClassifyItemResponse(
                    index=idx,
                    model_version=request.model_version or MODEL_VERSION,
                    error=str(exc),
                )
            )
            failed += 1

    return BatchClassifyResponse(
        processed=len(request.logs),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.post("/offline/train", response_model=OfflineTrainResponse)
def offline_train(request: OfflineTrainRequest):
    model_version = request.model_version or MODEL_VERSION
    detail = (
        "Offline training stub: connect to your training pipeline "
        "to fit vectorizer/encoders, clusterer, and classifier."
    )
    return OfflineTrainResponse(
        model_version=model_version,
        status="accepted",
        detail=detail,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
