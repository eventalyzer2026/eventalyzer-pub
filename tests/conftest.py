import os
import time
from typing import Optional

import pytest
import requests


def _is_up(url: str) -> bool:
    try:
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


def _has_indices(es_url: str, pattern: str) -> bool:
    try:
        resp = requests.get(f"{es_url}/_cat/indices/{pattern}?h=index", timeout=5)
        if resp.status_code >= 400:
            return False
        return bool(resp.text.strip())
    except Exception:
        return False


@pytest.fixture(scope="session")
def es_url() -> str:
    return os.getenv("ES_URL", "http://localhost:9200")


@pytest.fixture(scope="session")
def es_index_pattern() -> str:
    # Default to logs-raw-* to match Logstash config; override via env if needed.
    return os.getenv("ES_INDEX_PATTERN", "raw-logs-*")


@pytest.fixture(scope="session")
def ml_url() -> str:
    return os.getenv("ML_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def ml_available(ml_url) -> bool:
    return _is_up(f"{ml_url}/health")


@pytest.fixture(scope="session")
def es_available(es_url) -> bool:
    return _is_up(es_url)


def poll_es_count(es_url: str, index_pattern: str, timeout: int = 30, initial: Optional[int] = None) -> int:
    """Poll Elasticsearch _count until it changes or timeout is hit."""
    end_time = time.time() + timeout
    last = initial
    while time.time() < end_time:
        try:
            resp = requests.get(
                f"{es_url}/{index_pattern}/_count",
                timeout=5,
            )
            if resp.status_code == 404:
                last = 0
            else:
                resp.raise_for_status()
                last = resp.json().get("count", 0)
            if initial is None or last > initial:
                return last
        except Exception:
            pass
        time.sleep(1)
    return last if last is not None else 0


@pytest.fixture(scope="session")
def es_has_indices(es_url, es_index_pattern) -> bool:
    return _has_indices(es_url, es_index_pattern)
