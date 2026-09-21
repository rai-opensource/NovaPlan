#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import unittest

from novaplan.flow_extraction_client import FlowExtractionClient


class _FakeResponse:
    def __init__(self, *, headers=None, json_data=None):
        self.headers = headers or {}
        self._json_data = json_data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url, **kwargs):
        del kwargs
        self.urls.append(url)
        if len(self.urls) > 1:
            raise AssertionError(f"Unexpected GET {url}")
        return self.response


class VideoFlowExtractionClientTest(unittest.TestCase):
    def test_selection_flow_contract_accepts_current_video_server(self):
        client = FlowExtractionClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            _FakeResponse(
                json_data={
                    "contract_version": 3,
                    "wan_full_inline_flow": True,
                    "flow_only_available": True,
                },
            )
        )

        client.ensure_selection_flow_contract()
        client.ensure_selection_flow_contract()

        self.assertEqual(client._session.urls, ["http://video-server/health"])

    def test_selection_flow_contract_rejects_stale_video_server(self):
        client = FlowExtractionClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            _FakeResponse(
                json_data={
                    "contract_version": 2,
                    "wan_full_inline_flow": True,
                },
            )
        )

        with self.assertRaisesRegex(RuntimeError, "stale"):
            client.ensure_selection_flow_contract()

    def test_fetch_outputs_does_not_probe_metric_flow_endpoints(self):
        client = FlowExtractionClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            _FakeResponse(
                headers={"content-type": "application/json"},
                json_data={"flow_images": [{"data_base64": "cG5n"}]},
            )
        )

        outputs = client._fetch_flow_outputs("job-123")

        self.assertEqual(outputs["flow_images"], [b"png"])
        self.assertEqual(outputs["coords_3d"], [])
        self.assertEqual(outputs["visibilities"], [])
        self.assertEqual(client._session.urls, ["http://video-server/result/job-123/all"])


if __name__ == "__main__":
    unittest.main()
