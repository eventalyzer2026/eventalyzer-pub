import os
import time

import pytest
import requests

from conftest import poll_es_count


SAMPLE_LINE = '188.94.33.68 - - [09/Dec/2025:00:00:05 +0300] "POST /mod/quiz/processattempt.php?cmid=283 HTTP/1.1" 303 4727 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"'


@pytest.mark.integration
def test_log_ingestion_increments_count(es_available, es_url, es_index_pattern):
    if not es_available:
        pytest.skip("Elasticsearch is not reachable")

    # Initial count
    start_count = poll_es_count(es_url, es_index_pattern, timeout=5)

    # Append one log line to the monitored file path (host path mounted into Filebeat/Logstash)
    log_path = os.getenv("LOG_PATH", "logs/moodle_access.log")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(SAMPLE_LINE + "\n")

    # Allow Filebeat/Logstash time to ingest
    new_count = poll_es_count(es_url, es_index_pattern, timeout=30, initial=start_count)

    assert new_count > start_count, f"Expected count to increase after ingest, was {start_count}, now {new_count}"
