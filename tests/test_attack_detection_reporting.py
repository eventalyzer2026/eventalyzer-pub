from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest
import requests


ATTACK_EXAMPLES: list[dict[str, str]] = [
    {
        "attack_class": "xss_probe",
        "mitre_technique": "T1059.007",
        "method": "GET",
        "path": "/comment?text=%3Cimg+src%3Dx+onerror%3Dalert%281%29%3E",
    },
    {
        "attack_class": "bruteforce_probe",
        "mitre_technique": "T1110",
        "method": "POST",
        "path": "/login?username=admin&password=letmein",
    },
    {
        "attack_class": "deserialization_probe",
        "mitre_technique": "T1190",
        "method": "GET",
        "path": "/api/session?blob=O%3A8%3A%22stdClass%22%3A1%3A%7Bs%3A4%3A%22test%22%3Bs%3A4%3A%22boom%22%3B%7D",
    },
    {
        "attack_class": "ssti_probe",
        "mitre_technique": "T1190",
        "method": "GET",
        "path": "/render?tpl=%7B%7B7%2A7%7D%7D",
    },
    {
        "attack_class": "scanner_probe",
        "mitre_technique": "T1595.002",
        "method": "GET",
        "path": "/.env",
    },
    {
        "attack_class": "open_redirect_probe",
        "mitre_technique": "T1190",
        "method": "GET",
        "path": "/logout?returnTo=http%3A%2F%2Fevil.example%2Flanding",
    },
    {
        "attack_class": "xxe_probe",
        "mitre_technique": "T1190",
        "method": "POST",
        "path": "/api/xml?xml=%3C%21DOCTYPE+foo+%5B%3C%21ENTITY+x+SYSTEM+%22file%3A%2F%2F%2Fetc%2Fpasswd%22%3E%5D%3E",
    },
    {
        "attack_class": "lfi_probe",
        "mitre_technique": "T1006",
        "method": "GET",
        "path": "/index.php?page=..%2F..%2F..%2F..%2Fetc%2Fpasswd",
    },
    {
        "attack_class": "path_traversal_probe",
        "mitre_technique": "T1006",
        "method": "GET",
        "path": "/download?file=..%25252f..%25252f..%25252fetc%25252fshadow",
    },
    {
        "attack_class": "sqli_probe",
        "mitre_technique": "T1190",
        "method": "GET",
        "path": "/login?username=admin%27--&password=x",
    },
    {
        "attack_class": "rce_webshell_probe",
        "mitre_technique": "T1505.003",
        "method": "GET",
        "path": "/cmd.php?exec=whoami",
    },
    {
        "attack_class": "command_injection_probe",
        "mitre_technique": "T1059",
        "method": "GET",
        "path": "/ping?host=127.0.0.1%7Ccat+%2Fetc%2Fpasswd",
    },
]


def _now_utc_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_attack_ecs(example: dict[str, str]) -> dict[str, Any]:
    path = example["path"]
    method = example["method"]
    ts = _now_utc_z()
    clean_path = path.split("?", 1)[0]

    return {
        "source": {"ip": "127.0.0.1"},
        "type": "apache_access",
        "@timestamp": ts,
        "event": {
            "original": f'127.0.0.1 - - [{ts}] "{method} {path} HTTP/1.1" 404 0 "-" "EventalyzerAttackTests/1.0"',
            "type": ["access", "connection"],
            "dataset": "apache.access",
            "module": "apache",
            "category": ["web", "intrusion_detection"],
            "kind": "event",
        },
        "host": {"name": "eventalyzer-attack-test"},
        "message": f"{method} {path}",
        "http": {
            "response": {"status_code": 404, "body": {"bytes": 0}},
            "version": "1.1",
            "request": {"referrer": "-", "method": method},
        },
        "url": {"original": path, "path": clean_path},
        "user_agent": {
            "original": "EventalyzerAttackTests/1.0",
            "name": "EventalyzerAttackTests",
            "version": "1.0",
            "os": {"name": "Linux", "version": "unknown", "full": "Linux"},
            "device": {"name": "Other"},
        },
        "ecs": {"version": "8.0.0"},
        "log": {"file": {"path": "/var/log/apache/access.log"}},
    }


def _report_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "test_reports"


def _write_report(filename_prefix: str, payload: dict[str, Any]) -> Path:
    report_dir = _report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{filename_prefix}_{ts}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _failure_preview(failures: list[dict[str, Any]], limit: int = 3) -> str:
    rows = []
    for item in failures[:limit]:
        rows.append(
            f"class={item.get('attack_class')} method={item.get('method')} path={item.get('path')} reason={item.get('reason')}"
        )
    return " | ".join(rows)


@pytest.mark.integration
def test_attack_examples_detected_as_anomaly_with_report(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for example in ATTACK_EXAMPLES:
        ecs_log = _build_attack_ecs(example)

        entry: dict[str, Any] = {
            "attack_class": example["attack_class"],
            "mitre_technique": example["mitre_technique"],
            "method": example["method"],
            "path": example["path"],
            "request": ecs_log,
        }

        try:
            resp = requests.post(f"{ml_url}/classify", json=ecs_log, timeout=30.0)
            entry["status_code"] = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            entry["response"] = body
        except Exception as exc:
            entry["status_code"] = None
            entry["response"] = {"error": str(exc)}
            entry["reason"] = f"request_error: {exc}"
            failures.append(entry)
            attempts.append(entry)
            continue

        if entry["status_code"] != 200:
            entry["reason"] = f"http_{entry['status_code']}"
            failures.append(entry)
        else:
            cluster = entry["response"].get("cluster")
            if cluster != -1:
                entry["reason"] = f"unexpected_cluster={cluster}"
                failures.append(entry)

        attempts.append(entry)

    report = {
        "test": "test_attack_examples_detected_as_anomaly_with_report",
        "total": len(attempts),
        "failed": len(failures),
        "failed_classes": sorted({f["attack_class"] for f in failures}),
        "attempts": attempts,
    }
    report_path = _write_report("attack_detection_single", report)

    assert not failures, (
        f"{len(failures)} attack examples were not detected as anomaly (cluster=-1). "
        f"Report: {report_path}. Preview: {_failure_preview(failures)}"
    )


@pytest.mark.integration
def test_batch_attack_examples_detected_as_anomaly_with_report(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    expanded_examples = ATTACK_EXAMPLES * 2
    logs = [_build_attack_ecs(example) for example in expanded_examples]

    req_payload = {"logs": logs, "return_proba": True}

    try:
        resp = requests.post(f"{ml_url}/batch/classify", json=req_payload, timeout=60.0)
    except Exception as exc:
        pytest.fail(f"Batch request failed: {exc}")

    body: dict[str, Any]
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    failures: list[dict[str, Any]] = []
    result_by_index = {item.get("index"): item for item in body.get("results", []) if isinstance(item, dict)}

    for idx, example in enumerate(expanded_examples):
        result = result_by_index.get(idx)
        failure: dict[str, Any] | None = None
        
        if resp.status_code != 200:
            failure = {
                "index": idx,
                "attack_class": example["attack_class"],
                "method": example["method"],
                "path": example["path"],
                "reason": f"http_{resp.status_code}",
                "result": body,
            }
        elif result is None:
            failure = {
                "index": idx,
                "attack_class": example["attack_class"],
                "method": example["method"],
                "path": example["path"],
                "reason": "missing_result_item",
                "result": None,
            }
        elif result.get("error"):
            failure = {
                "index": idx,
                "attack_class": example["attack_class"],
                "method": example["method"],
                "path": example["path"],
                "reason": f"item_error={result.get('error')}",
                "result": result,
            }
        elif result.get("cluster") != -1:
            failure = {
                "index": idx,
                "attack_class": example["attack_class"],
                "method": example["method"],
                "path": example["path"],
                "reason": f"unexpected_cluster={result.get('cluster')}",
                "result": result,
            }

        if failure is not None:
            failures.append(failure)

    report = {
        "test": "test_batch_attack_examples_detected_as_anomaly_with_report",
        "request_size": len(logs),
        "status_code": resp.status_code,
        "processed": body.get("processed") if isinstance(body, dict) else None,
        "succeeded": body.get("succeeded") if isinstance(body, dict) else None,
        "failed": body.get("failed") if isinstance(body, dict) else None,
        "failed_items": failures,
        "response": body,
    }
    report_path = _write_report("attack_detection_batch", report)

    assert resp.status_code == 200, f"Batch classify returned HTTP {resp.status_code}. Report: {report_path}"
    assert body.get("processed") == len(logs), f"Unexpected processed count: {body.get('processed')}. Report: {report_path}"
    assert not failures, (
        f"{len(failures)} batch attack items were not detected as anomaly (cluster=-1). "
        f"Report: {report_path}. Preview: {_failure_preview(failures)}"
    )
