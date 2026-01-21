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
BATCH_SIZE = int(os.getenv("AGENT_BATCH_SIZE", "10"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "current")


def classify_via_eventalyzer(payload: dict) -> tuple[int, Optional[float], Optional[str]]:
    resp = requests.post(
        f"{EVENTALYZER_URL}/online/predict",
        json={"log": payload, "return_proba": True},
        timeout=15,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Eventalyzer error {resp.status_code}: {resp.text}")
    body = resp.json()
    return int(body.get("cluster")), body.get("proba"), body.get("model_version")


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

        hits = search_raw_batch(ES_URL, RAW_INDEX_PATTERN, BATCH_SIZE, {"match_all": {}}, search_after)
        logging.info("Done first request, ans len: %d", len(hits))
        if not hits:
            time.sleep(POLL_INTERVAL)
            continue

        docs_for_index = []
        for hit in hits:
            source = hit.get("_source", {})
            url_obj = source.get("url", {}) if isinstance(source.get("url"), dict) else {}
            url_path = url_obj.get("path")
            url_original = url_obj.get("original")
            request_path = source.get("request")
            if not url_path and (url_original or request_path):
                derived = (url_original or request_path)
                if isinstance(derived, str):
                    derived = derived.split("?", 1)[0]
                source.setdefault("url", {})
                source["url"]["path"] = derived
            if not source.get("url", {}).get("path"):
                logging.info("Skipping doc %s: missing url.path", hit.get("_id"))
                logging.info("Event: %s", source)
                continue
            try:
                cluster_id, proba, model_version = classify_via_eventalyzer(source)
                ts_value = source.get("@timestamp")
                result_doc = {
                    "@timestamp": ts_value,
                    "source_doc_id": hit.get("_id"),
                    "source_index": hit.get("_index"),
                    "ml": {
                        "cluster_id": cluster_id,
                        "proba": proba,
                        "model_version": model_version or MODEL_VERSION,
                        "is_anomaly": cluster_id == -1,
                    },
                }
                docs_for_index.append((result_doc, hit.get("_id"), ts_value))
                logging.info("Prepared result for %s", hit.get("_id"))
            except Exception as exc:
                logging.error("Failed to classify doc %s: %s; ORIG MSG: %s", hit.get("_id"), exc, json.dumps(hit))
                continue

        indexed, errors = bulk_index_results(ES_URL, RESULTS_INDEX_PREFIX, docs_for_index)
        if errors:
            logging.warning("Indexed %d docs with errors: %s", indexed, errors[:3])
        else:
            logging.info("Indexed %d docs", indexed)

        # Update cursor for next batch
        last_sort = hits[-1].get("sort")
        if last_sort:
            search_after = last_sort
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()
