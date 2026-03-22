from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from backend.app.services.data_client import DataClient


class DataClientRetryTests(unittest.TestCase):
    @staticmethod
    def _resp(status_code: int, payload=None):
        r = Mock()
        r.status_code = int(status_code)
        r.json.return_value = payload or {}
        r.raise_for_status.return_value = None
        return r

    def test_request_json_retries_on_connection_error_then_succeeds(self):
        client = DataClient(timeout_seconds=1, cache_ttl_seconds=0, max_retries=2, backoff_base_seconds=0.01)
        ok = self._resp(200, {"ok": True})
        with (
            patch("backend.app.services.data_client.requests.get", side_effect=[requests.ConnectionError("boom"), ok]) as get_mock,
            patch("backend.app.services.data_client.time.sleep") as sleep_mock,
        ):
            out = client._request_json("https://example.test", {"a": 1}, "test source")
        self.assertEqual(out, {"ok": True})
        self.assertEqual(get_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_request_json_retries_on_retryable_http_status(self):
        client = DataClient(timeout_seconds=1, cache_ttl_seconds=0, max_retries=2, backoff_base_seconds=0.01)
        retryable = self._resp(502, {})
        ok = self._resp(200, {"hourly": {"time": []}})
        with (
            patch("backend.app.services.data_client.requests.get", side_effect=[retryable, ok]) as get_mock,
            patch("backend.app.services.data_client.time.sleep") as sleep_mock,
        ):
            out = client._request_json("https://example.test", {"b": 2}, "test source")
        self.assertEqual(out, {"hourly": {"time": []}})
        self.assertEqual(get_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_request_json_does_not_retry_non_retryable_http_status(self):
        client = DataClient(timeout_seconds=1, cache_ttl_seconds=0, max_retries=2, backoff_base_seconds=0.01)
        bad = self._resp(400, {})
        bad.raise_for_status.side_effect = requests.HTTPError("400 bad request", response=bad)
        with (
            patch("backend.app.services.data_client.requests.get", return_value=bad) as get_mock,
            patch("backend.app.services.data_client.time.sleep") as sleep_mock,
        ):
            with self.assertRaises(RuntimeError):
                client._request_json("https://example.test", {"c": 3}, "test source")
        self.assertEqual(get_mock.call_count, 1)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
