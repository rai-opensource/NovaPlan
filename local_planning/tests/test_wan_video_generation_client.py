#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.video_generation import WanVideoGenerationClient  # noqa: E402


class _FakeResponse:
    def __init__(self, *, status_code=200, headers=None, content=b"", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **kwargs):
        del kwargs
        self.urls.append(url)
        if not self.responses:
            raise AssertionError(f"Unexpected GET {url}")
        return self.responses.pop(0)


class WanVideoGenerationClientTest(unittest.TestCase):
    def test_wan_full_contract_preflight_accepts_current_server(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession([
            _FakeResponse(json_data={"contract_version": 3, "wan_full_inline_flow": True}),
        ])

        client._ensure_wan_full_contract()
        client._ensure_wan_full_contract()

        self.assertEqual(client._session.urls, ["http://video-server/health"])

    def test_wan_full_contract_preflight_rejects_stale_server(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession([_FakeResponse(status_code=404)])

        with self.assertRaisesRegex(RuntimeError, "required for WAN rollout generation is unreachable"):
            client._ensure_wan_full_contract()

    def test_wan_full_contract_preflight_rejects_version_two_server(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession([
            _FakeResponse(json_data={"contract_version": 2, "wan_full_inline_flow": True}),
        ])

        with self.assertRaisesRegex(RuntimeError, "running video-generation server is stale"):
            client._ensure_wan_full_contract()

    def test_wan_full_mode_requires_inline_flow(self):
        with self.assertRaisesRegex(RuntimeError, "WAN full-mode jobs returned video data"):
            WanVideoGenerationClient._complete_output_indices(
                [object(), object()],
                [None, None],
                require_inline_flow=True,
            )

    def test_wan_full_mode_keeps_only_video_flow_pairs(self):
        paired, missing = WanVideoGenerationClient._complete_output_indices(
            [object(), object(), None],
            [object(), None, object()],
            require_inline_flow=True,
        )
        self.assertEqual(paired, [0])
        self.assertEqual(missing, [1])

    def test_generate_only_does_not_require_flow(self):
        paired, missing = WanVideoGenerationClient._complete_output_indices(
            [object(), object()],
            [None, None],
            require_inline_flow=False,
        )
        self.assertEqual(paired, [0, 1])
        self.assertEqual(missing, [])

    def test_fetch_result_bytes_prefers_binary_endpoint(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            [
                _FakeResponse(headers={"content-type": "video/mp4"}, content=b"mp4-bytes"),
            ]
        )

        result = client._fetch_result_bytes("job-123")

        self.assertEqual(result, [b"mp4-bytes"])
        self.assertEqual(client._session.urls, ["http://video-server/result/job-123/binary"])

    def test_fetch_result_bytes_falls_back_to_json_result(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            [
                _FakeResponse(status_code=404),
                _FakeResponse(
                    headers={"content-type": "application/json"},
                    json_data={"video_base64": "bXA0LWJ5dGVz"},
                ),
            ]
        )

        result = client._fetch_result_bytes("job-123")

        self.assertEqual(result, [b"mp4-bytes"])
        self.assertEqual(
            client._session.urls,
            [
                "http://video-server/result/job-123/binary",
                "http://video-server/result/job-123",
            ],
        )

    def test_fetch_all_results_does_not_probe_metric_flow_endpoints(self):
        client = WanVideoGenerationClient(server_base="http://video-server", timeout=1)
        client._session = _FakeSession(
            [
                _FakeResponse(
                    headers={"content-type": "application/json"},
                    json_data={
                        "videos": [{"data_base64": "bXA0"}],
                        "flow_images": [{"data_base64": "cG5n"}],
                    },
                ),
            ]
        )

        videos, flows, coords_3d, visibilities = client._fetch_all_results("job-123")

        self.assertEqual(videos, [b"mp4"])
        self.assertEqual(flows, [b"png"])
        self.assertEqual(coords_3d, [])
        self.assertEqual(visibilities, [])
        self.assertEqual(client._session.urls, ["http://video-server/result/job-123/all"])


if __name__ == "__main__":
    unittest.main()
