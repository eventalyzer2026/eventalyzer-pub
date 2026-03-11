import logging
import os
import time
import json
from typing import Any, List, Optional

import requests

from es_client import bulk_index_results, ping, search_raw_batch

logging.basicConfig(level=logging.INFO)

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
EVENTALYZER_URL = os.getenv("EVENTALYZER_URL", "http://eventalyzer:8000")
RAW_INDEX_PATTERN = os.getenv("RAW_INDEX_PATTERN", "raw-logs-*")
RESULTS_INDEX_PREFIX = os.getenv("RESULTS_INDEX_PREFIX", "ml-results")
POLL_INTERVAL = float(os.getenv("AGENT_POLL_INTERVAL", "5"))
BATCH_SIZE = int(os.getenv("AGENT_BATCH_SIZE", "100"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "current")


def classify_batch_via_eventalyzer(payloads: List[dict]) -> dict:
    resp = requests.post(
        f"{EVENTALYZER_URL}/batch/classify",
        json={"logs": payloads, "return_proba": True},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Eventalyzer batch error {resp.status_code}: {resp.text}")
    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError("Eventalyzer batch response is not an object")
    return body


def _normalize_for_model(source: dict) -> Optional[dict]:
    url_obj = source.get("url", {}) if isinstance(source.get("url"), dict) else {}
    url_path = url_obj.get("path")
    url_original = url_obj.get("original")
    request_path = source.get("request")

    if not url_path and (url_original or request_path):
        derived = url_original or request_path
        if isinstance(derived, str):
            derived = derived.split("?", 1)[0]
        source.setdefault("url", {})
        source["url"]["path"] = derived

    if not source.get("url", {}).get("path"):
        return None

    return source


def run_loop():
    logging.info(
        "Starting agent: ES=%s, eventalyzer=%s, raw=%s, results=%s",
        ES_URL,
        EVENTALYZER_URL,
        RAW_INDEX_PATTERN,
        RESULTS_INDEX_PREFIX,
    )
    search_after: Optional[List[Any]] = None

    while True:
        if not ping(ES_URL):
            logging.warning("Elasticsearch not reachable, retrying...")
            time.sleep(POLL_INTERVAL)
            continue
        
        t1 = time.time()
        hits = search_raw_batch(ES_URL, RAW_INDEX_PATTERN, BATCH_SIZE, {"match_all": {}}, search_after)
        t2 = time.time()
        logging.info("Fetched batch from ES, size=%d, estimated: %f", len(hits), t2 - t1)

        if not hits:
            time.sleep(POLL_INTERVAL)
            continue

        valid_hits: List[dict] = []
        payloads: List[dict] = []

        for hit in hits:
            source = hit.get("_source", {})
            normalized = _normalize_for_model(source)
            if normalized is None:
                logging.info("Skipping doc %s: missing url.path", hit.get("_id"))
                logging.info("Event: %s", source)
                continue
            valid_hits.append(hit)
            payloads.append(normalized)

        if not payloads:
            last_sort = hits[-1].get("sort")
            if last_sort:
                search_after = last_sort
            time.sleep(POLL_INTERVAL)
            continue

        docs_for_index = []

        try:
            batch_result = classify_batch_via_eventalyzer(payloads)
            items = batch_result.get("results", [])
            if not isinstance(items, list):
                raise RuntimeError("Eventalyzer batch response does not contain result list")

            for item in items:
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(valid_hits):
                    logging.error("Invalid batch result index: %s", json.dumps(item))
                    continue

                hit = valid_hits[idx]
                source = hit.get("_source", {})
                ts_value = source.get("@timestamp")
                error = item.get("error")
                if error:
                    logging.error("Model failed for doc %s: %s", hit.get("_id"), error)
                    continue

                cluster_id = item.get("cluster")
                result_doc = {
                    "@timestamp": ts_value,
                    "source_doc_id": hit.get("_id"),
                    "source_index": hit.get("_index"),
                    "ml": {
                        "cluster_id": cluster_id,
                        "proba": item.get("proba"),
                        "model_version": item.get("model_version") or MODEL_VERSION,
                        "is_anomaly": cluster_id == -1,
                    },
                    "event": {
                        "original": (source.get("event") or {}).get("original") if isinstance(source.get("event"), dict) else None,
                    },
                }
                docs_for_index.append((result_doc, hit.get("_id"), ts_value))

        except Exception as exc:
            logging.error("Batch classification failed: %s", exc)
            time.sleep(POLL_INTERVAL)
            continue

        indexed, errors = bulk_index_results(ES_URL, RESULTS_INDEX_PREFIX, docs_for_index)

        if errors:
            logging.warning("Indexed %d docs with errors: %s", indexed, errors[:3])
        else:
            logging.info("Indexed %d docs", indexed)

        last_sort = hits[-1].get("sort")
        if last_sort:
            search_after = last_sort
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()
