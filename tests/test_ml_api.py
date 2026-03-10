import concurrent.futures
import math
import time

import pytest
import requests


SAMPLE_ECS_LOG = {
  "source": {
    "ip": "31.185.5.73"
  },
  "type": "apache_access",
  "@timestamp": "2025-12-08T22:32:13.000Z",
  "event": {
    "original": "31.185.5.73 - - [09/Dec/2025:01:32:13 +0300] \"GET /lib/javascript.php/1758485667/lib/requirejs/require.min.js HTTP/1.1\" 200 9685 \"https://it548.ru/login/index.php\" \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 YaBrowser/25.10.0.0 Safari/537.36\"",
    "type": [
      "access",
      "connection"
    ],
    "dataset": "apache.access",
    "module": "apache",
    "category": [
      "web",
      "authentication"
    ],
    "kind": "event"
  },
  "host": {
    "name": "e9828ba7e444"
  },
  "message": "31.185.5.73 - - [09/Dec/2025:01:32:13 +0300] \"GET /lib/javascript.php/1758485667/lib/requirejs/require.min.js HTTP/1.1\" 200 9685 \"https://it548.ru/login/index.php\" \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 YaBrowser/25.10.0.0 Safari/537.36\"",
  "http": {
    "response": {
      "status_code": 200,
      "body": {
        "bytes": 9685
      }
    },
    "version": "1.1",
    "request": {
      "referrer": "https://it548.ru/login/index.php",
      "method": "GET"
    }
  },
  "url": {
    "original": "/lib/javascript.php/1758485667/lib/requirejs/require.min.js",
    "path": "/lib/javascript.php/1758485667/lib/requirejs/require.min.js"
  },
  "user_agent": {
    "original": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 YaBrowser/25.10.0.0 Safari/537.36",
    "version": "25.10.0.0",
    "os": {
      "version": "10",
      "full": "Windows 10",
      "name": "Windows"
    },
    "device": {
      "name": "Other"
    },
    "name": "Yandex Browser"
  },
  "ecs": {
    "version": "8.0"
  },
  "log": {
    "file": {
      "path": "/var/log/test_logs/moodle_access.log"
    }
  }
}


def _classify(ml_url: str, payload: dict, timeout: float = 10.0):
    resp = requests.post(f"{ml_url}/classify", json=payload, timeout=timeout)
    return resp


def _assert_valid_response(resp_json: dict):
    assert isinstance(resp_json.get("cluster"), int)
    assert resp_json["cluster"] >= -1
    proba = resp_json.get("proba")
    assert proba is None or (isinstance(proba, (int, float)) and 0.0 <= proba <= 1.0)
    if resp_json["cluster"] == -1:
        assert resp_json.get("detail") == "Event is anomaly"


@pytest.mark.integration
def test_classify_success(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    resp = _classify(ml_url, SAMPLE_ECS_LOG)
    assert resp.status_code == 200, f"Status {resp.status_code}, body: {resp.text}"
    data = resp.json()
    _assert_valid_response(data)


@pytest.mark.integration
def test_classify_deterministic(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    first_resp = _classify(ml_url, SAMPLE_ECS_LOG)
    second_resp = _classify(ml_url, SAMPLE_ECS_LOG)
    assert first_resp.status_code == 200, f"First status {first_resp.status_code}, body: {first_resp.text}"
    assert second_resp.status_code == 200, f"Second status {second_resp.status_code}, body: {second_resp.text}"
    first = first_resp.json()
    second = second_resp.json()
    assert first["cluster"] == second["cluster"]
    # allow minor float rounding differences
    if first.get("proba") is not None and second.get("proba") is not None:
        assert math.isclose(first["proba"], second["proba"], rel_tol=1e-3, abs_tol=1e-3)


@pytest.mark.integration
def test_classify_invalid_payload(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    bad_payload = {k: v for k, v in SAMPLE_ECS_LOG.items() if k != "url"}  # missing url fields
    resp = _classify(ml_url, bad_payload)
    assert resp.status_code >= 400, f"Expected 4xx, got {resp.status_code}, body: {resp.text}"


@pytest.mark.integration
def test_classify_load_100rps(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    total_requests = 100
    max_workers = 50
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_classify, ml_url, SAMPLE_ECS_LOG, 90.0) for _ in range(total_requests)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                resp = fut.result()
                if resp.status_code >= 500:
                    failures.append((resp.status_code, resp.text))
            except Exception as exc:
                failures.append((None, str(exc)))

    assert not failures, f"Got failures under load: {failures}"


@pytest.mark.integration
def test_batch_classify(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    payload = {
        "logs": [SAMPLE_ECS_LOG] * 5,
        "return_proba": True,
    }

    resp = requests.post(f"{ml_url}/batch/classify", json=payload, timeout=30.0)
    assert resp.status_code == 200, f"Status {resp.status_code}, body: {resp.text}"

    data = resp.json()
    assert data.get("processed") == 5
    assert isinstance(data.get("failed"), int)
    assert isinstance(data.get("results"), list)
    assert len(data.get("results")) == 5

    print(f"Batch classify errors: {data.get('failed', 0)}")
