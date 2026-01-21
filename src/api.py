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
)
from app.src.classificator import LogClassificatorModel
from app.src.clusterer import LogClustererModel
from app.src.vertorized import LogVertorizer
from app.src.es_client import bulk_index_results, search_raw, ping

logging.basicConfig(level=logging.INFO)
MODEL_VERSION = os.getenv("MODEL_VERSION", "current")
ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
RAW_INDEX_PATTERN = os.getenv("RAW_INDEX_PATTERN", "raw-logs-*")
RESULTS_INDEX_PREFIX = os.getenv("RESULTS_INDEX_PREFIX", "ml-results")

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

    ass.classifier = LogClassificatorModel("app/src/models/log_classifier.joblib")
    ass.clusterer = LogClustererModel("app/src/models/clusterer.joblib")
    ass.es_available = ping(ES_URL)
    if ass.es_available:
        logging.info("Connected to Elasticsearch at %s", ES_URL)
    else:
        logging.warning("Elasticsearch is not reachable at %s", ES_URL)

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
        elapsed=t2-t1,
        detail=detail
    )


@app.post("/online/predict", response_model=OnlinePredictResponse)
def online_predict(request: OnlinePredictRequest):
    response = _predict_internal(
        log=request.log,
        return_proba=request.return_proba,
        model_version=request.model_version,
    )
    if request.write_to_es:
        if not ass.es_available:
            raise HTTPException(status_code=503, detail="Elasticsearch is not available")
        doc = {
            "@timestamp": request.log.model_dump().get("@timestamp"),
            "source_doc_id": request.source_doc_id,
            "ml": {
                "cluster_id": response.cluster,
                "proba": response.proba,
                "model_version": response.model_version,
                "is_anomaly": response.cluster == -1,
            },
        }
        prefix = request.results_index_prefix or RESULTS_INDEX_PREFIX
        bulk_index_results(
            ES_URL,
            prefix,
            [(doc, request.source_doc_id, doc.get("@timestamp"))],
        )
    return response


@app.post("/classify", response_model=OnlinePredictResponse)
def classify(log: ECSLog):
    return _predict_internal(log=log, return_proba=True)


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
        detail=detail
    )


@app.post("/batch/classify", response_model=BatchClassifyResponse)
def batch_classify(request: BatchClassifyRequest):
    if ass.vectorizer is None or ass.classifier is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")
    if not ass.es_available:
        raise HTTPException(status_code=503, detail="Elasticsearch is not available")

    raw_index = request.raw_index_pattern or RAW_INDEX_PATTERN
    results_prefix = request.results_index_prefix or RESULTS_INDEX_PREFIX
    query = request.query or {"match_all": {}}

    hits = search_raw(ES_URL, raw_index, request.size, query)
    docs_for_index = []
    errors = []

    for hit in hits:
        logging.info(hit.text)
        source = hit.get("_source", {})
        try:
            log = ECSLog.model_validate(source)
            response = _predict_internal(log=log, return_proba=True)
            ts_value = source.get("@timestamp")
            result_doc = {
                "@timestamp": ts_value,
                "source_doc_id": hit.get("_id"),
                "source_index": hit.get("_index"),
                "ml": {
                    "cluster_id": response.cluster,
                    "proba": response.proba,
                    "model_version": response.model_version,
                    "is_anomaly": response.cluster == -1,
                },
            }
            docs_for_index.append((result_doc, hit.get("_id"), ts_value))
        except Exception as exc:
            errors.append(str(exc))

    indexed, index_errors = bulk_index_results(ES_URL, results_prefix, docs_for_index)
    errors.extend(index_errors)

    return BatchClassifyResponse(
        processed=len(hits),
        indexed=indexed,
        errors=errors or None,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
