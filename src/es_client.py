import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, Optional
import os
import requests

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")


def _request(method: str, url: str, **kwargs) -> requests.Response:
    resp = requests.request(method, url, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp


def ping(es_url: str) -> bool:
    try:
        resp = requests.get(es_url, timeout=5)
        return resp.status_code < 400
    except Exception:
        return False


def search_raw(es_url: str, index_pattern: str, size: int, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = {
        "size": size,
        "query": query or {"match_all": {}},
        "sort": [{"@timestamp": {"order": "asc", "unmapped_type": "date", "missing": "_last"}}],
    }
    try:
        resp = _request(
            "POST",
            f"{es_url}/{index_pattern}/_search?ignore_unavailable=true&allow_no_indices=true",
            json=payload,
        )
        return resp.json().get("hits", {}).get("hits", [])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 404):
            logging.warning("search_raw failed: %s", exc.response.text)
            return []
        raise


def search_raw_batch(
    es_url: str,
    index_pattern: str,
    size: int,
    query: Dict[str, Any],
    search_after: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    payload = {
        "size": size,
        "query": query or {"match_all": {}},
        "sort": [{"@timestamp": {"order": "asc", "unmapped_type": "date", "missing": "_last"}}],
    }
    if search_after:
        payload["search_after"] = search_after
    try:
        resp = _request(
            "POST",
            f"{es_url}/{index_pattern}/_search?ignore_unavailable=true&allow_no_indices=true",
            json=payload,
        )
        return resp.json().get("hits", {}).get("hits", [])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 404):
            logging.warning("search_raw_batch failed: %s", exc.response.text)
            return []
        raise


def _index_name(prefix: str, ts_value: str | None) -> str:
    if not ts_value:
        date_part = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        return f"{prefix}-{date_part}"
    try:
        # Normalize "Z" to "+00:00" for fromisoformat
        normalized = ts_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        date_part = dt.strftime("%Y.%m.%d")
    except ValueError:
        date_part = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    return f"{prefix}-{date_part}"


def bulk_index_results(
    es_url: str,
    results_prefix: str,
    docs: Iterable[Tuple[Dict[str, Any], str | None, str | None]],
) -> Tuple[int, List[str]]:
    """
    docs: iterable of (document, source_doc_id, timestamp)
    """
    actions = []
    for doc, source_doc_id, ts_value in docs:
        index_name = _index_name(results_prefix, ts_value)
        meta = {"index": {"_index": index_name}}
        if source_doc_id:
            meta["index"]["_id"] = source_doc_id
        actions.append(json.dumps(meta))
        actions.append(json.dumps(doc))

    if not actions:
        return 0, []

    payload = "\n".join(actions) + "\n"
    resp = _request(
        "POST",
        f"{es_url}/_bulk",
        data=payload,
        headers={"Content-Type": "application/x-ndjson"},
    )
    body = resp.json()
    errors = []
    if body.get("errors"):
        for item in body.get("items", []):
            for _, result in item.items():
                err = result.get("error")
                if err:
                    errors.append(str(err))
    indexed = len(body.get("items", [])) - len(errors)
    return indexed, errors
