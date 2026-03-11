from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any

import pytest
import requests


ATTACK_EXAMPLES: list[dict[str, str]] = [
    {"attack_class": "xss_probe", "mitre_technique": "T1059.007", "method": "GET", "path": "/comment?text=%3Cimg+src%3Dx+onerror%3Dalert%281%29%3E"},
    {"attack_class": "bruteforce_probe", "mitre_technique": "T1110", "method": "POST", "path": "/login?username=admin&password=letmein"},
    {"attack_class": "deserialization_probe", "mitre_technique": "T1190", "method": "GET", "path": "/api/session?blob=O%3A8%3A%22stdClass%22%3A1%3A%7Bs%3A4%3A%22test%22%3Bs%3A4%3A%22boom%22%3B%7D"},
    {"attack_class": "ssti_probe", "mitre_technique": "T1190", "method": "GET", "path": "/render?tpl=%7B%7B7%2A7%7D%7D"},
    {"attack_class": "scanner_probe", "mitre_technique": "T1595.002", "method": "GET", "path": "/.env"},
    {"attack_class": "open_redirect_probe", "mitre_technique": "T1190", "method": "GET", "path": "/logout?returnTo=http%3A%2F%2Fevil.example%2Flanding"},
    {"attack_class": "xxe_probe", "mitre_technique": "T1190", "method": "POST", "path": "/api/xml?xml=%3C%21DOCTYPE+foo+%5B%3C%21ENTITY+x+SYSTEM+%22file%3A%2F%2F%2Fetc%2Fpasswd%22%3E%5D%3E"},
    {"attack_class": "lfi_probe", "mitre_technique": "T1006", "method": "GET", "path": "/index.php?page=..%2F..%2F..%2F..%2Fetc%2Fpasswd"},
    {"attack_class": "path_traversal_probe", "mitre_technique": "T1006", "method": "GET", "path": "/download?file=..%25252f..%25252f..%25252fetc%25252fshadow"},
    {"attack_class": "sqli_probe", "mitre_technique": "T1190", "method": "GET", "path": "/login?username=admin%27--&password=x"},
    {"attack_class": "rce_webshell_probe", "mitre_technique": "T1505.003", "method": "GET", "path": "/cmd.php?exec=whoami"},
    {"attack_class": "command_injection_probe", "mitre_technique": "T1059", "method": "GET", "path": "/ping?host=127.0.0.1%7Ccat+%2Fetc%2Fpasswd"},
]


SAFE_NON_ATTACK_PREFIXES = (
    "/lib/javascript.php/",
    "/theme/styles.php/",
    "/lib/requirejs.php/",
    "/mod/quiz/processattempt.php",
    "/course/search.php",
    "/course/info.php",
    "/mod/quiz/summary.php",
    "/theme/yui_combo.php",
    "/theme/font.php/",
    "/theme/image.php/boost/theme/",
    "/lib/ajax/service.php",
    "/lib/ajax/service-nologin.php",
)


def _now_utc_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_attack_ecs(method: str, path: str, status_code: int, referrer: str) -> dict[str, Any]:
    ts = _now_utc_z()
    clean_path = path.split("?", 1)[0]

    return {
        "source": {"ip": "127.0.0.1"},
        "type": "apache_access",
        "@timestamp": ts,
        "event": {
            "original": f'127.0.0.1 - - [{ts}] "{method} {path} HTTP/1.1" {status_code} 0 "{referrer}" "EventalyzerAttackTests/1.0"',
            "type": ["access", "connection"],
            "dataset": "apache.access",
            "module": "apache",
            "category": ["web", "intrusion_detection"],
            "kind": "event",
        },
        "host": {"name": "eventalyzer-attack-test"},
        "message": f"{method} {path}",
        "http": {
            "response": {"status_code": status_code, "body": {"bytes": 0}},
            "version": "1.1",
            "request": {"referrer": referrer, "method": method},
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


def _load_non_attack_pool() -> list[dict[str, Any]]:
    default_path = Path(__file__).resolve().parents[1] / "parsed_logs" / "access_2025.12.08.json"
    logs_path = Path(os.getenv("EVENTALYZER_NON_ATTACK_LOGS_PATH", str(default_path)))
    if not logs_path.exists():
        pytest.skip(f"Non-attack source logs not found: {logs_path}")

    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for line in logs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        http = event.get("http") or {}
        req = http.get("request") or {}
        resp = http.get("response") or {}
        url = event.get("url") or {}

        path = str(url.get("original") or url.get("path") or "")
        status = resp.get("status_code")
        referrer = str(req.get("referrer") or "-")

        if not path:
            continue
        if status not in (200, 303):
            continue
        if not any(path.startswith(prefix) for prefix in SAFE_NON_ATTACK_PREFIXES):
            continue
        if not (referrer == "-" or "it548.ru" in referrer):
            continue
        if path in seen_paths:
            continue

        seen_paths.add(path)
        selected.append(event)

    if len(selected) < 5:
        pytest.fail(
            f"Not enough non-attack candidates in {logs_path}. "
            f"Found {len(selected)}, need at least 5."
        )
    return selected


def _report_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "test_reports"


def _write_report(prefix: str, payload: dict[str, Any]) -> Path:
    d = _report_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = d / f"{prefix}_{ts}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


@pytest.mark.integration
def test_random_5_attack_and_5_normal_strict_directional_asserts(ml_available, ml_url):
    if not ml_available:
        pytest.skip("ML service is not reachable")

    if len(ATTACK_EXAMPLES) < 5:
        pytest.fail("Not enough attack examples to sample 5 attack logs")

    env_seed = os.getenv("EVENTALYZER_TEST_SEED")
    seed = int(env_seed) if env_seed is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    rng = random.Random(seed)

    non_attack_pool = _load_non_attack_pool()
    picked_attacks = rng.sample(ATTACK_EXAMPLES, 5)
    picked_normals = rng.sample(non_attack_pool, 5)

    items: list[dict[str, Any]] = []

    for item in picked_attacks:
        items.append(
            {
                "group": "attack",
                "name": item["attack_class"],
                "method": item["method"],
                "path": item["path"],
                "expected_anomaly": True,
                "request": _build_attack_ecs(item["method"], item["path"], 404, "-"),
            }
        )

    for item in picked_normals:
        http = item.get("http") or {}
        req = http.get("request") or {}
        url = item.get("url") or {}
        method = str(req.get("method") or "GET")
        path = str(url.get("original") or url.get("path") or "")
        items.append(
            {
                "group": "normal",
                "name": "real_access_log",
                "method": method,
                "path": path,
                "expected_anomaly": False,
                "request": item,
            }
        )

    rng.shuffle(items)

    payload = {"logs": [x["request"] for x in items], "return_proba": True}
    resp = requests.post(f"{ml_url}/batch/classify", json=payload, timeout=60.0)

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    failures: list[dict[str, Any]] = []
    result_by_index = {entry.get("index"): entry for entry in body.get("results", []) if isinstance(entry, dict)}

    for idx, item in enumerate(items):
        result = result_by_index.get(idx)

        if resp.status_code != 200:
            failures.append({
                "index": idx,
                "group": item["group"],
                "name": item["name"],
                "path": item["path"],
                "reason": f"http_{resp.status_code}",
                "result": body,
            })
            continue

        if result is None:
            failures.append({
                "index": idx,
                "group": item["group"],
                "name": item["name"],
                "path": item["path"],
                "reason": "missing_result_item",
                "result": None,
            })
            continue

        if result.get("error"):
            failures.append({
                "index": idx,
                "group": item["group"],
                "name": item["name"],
                "path": item["path"],
                "reason": f"item_error={result.get('error')}",
                "result": result,
            })
            continue

        got_anomaly = result.get("cluster") == -1
        expected_anomaly = bool(item["expected_anomaly"])

        if expected_anomaly and not got_anomaly:
            failures.append({
                "index": idx,
                "group": item["group"],
                "name": item["name"],
                "path": item["path"],
                "reason": f"false_negative(cluster={result.get('cluster')})",
                "result": result,
            })

        if (not expected_anomaly) and got_anomaly:
            failures.append({
                "index": idx,
                "group": item["group"],
                "name": item["name"],
                "path": item["path"],
                "reason": f"false_positive(cluster={result.get('cluster')})",
                "result": result,
            })

    report_payload = {
        "test": "test_random_5_attack_and_5_normal_strict_directional_asserts",
        "seed": seed,
        "status_code": resp.status_code,
        "processed": body.get("processed") if isinstance(body, dict) else None,
        "picked_attacks": picked_attacks,
        "picked_normals": picked_normals,
        "mixed_items": [{k: v for k, v in item.items() if k != "request"} for item in items],
        "failures": failures,
        "response": body,
    }
    report_path = _write_report("attack_vs_normal_random", report_payload)

    assert resp.status_code == 200, f"Batch classify HTTP {resp.status_code}. seed={seed}. Report: {report_path}"
    assert body.get("processed") == 10, f"Expected processed=10, got {body.get('processed')}. seed={seed}. Report: {report_path}"
    assert not failures, f"Directional misclassifications={len(failures)}. seed={seed}. Report: {report_path}. Failures: {failures[:3]}"
