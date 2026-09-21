#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from remote_video_generation import video_generation_server as server  # noqa: E402


class VideoGenerationServerContractTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_output_dir = server.COMFYUI_OUTPUT_DIR
        self.original_jobs = server.jobs
        server.COMFYUI_OUTPUT_DIR = self.temp_dir.name
        server.jobs = {}

    def tearDown(self):
        server.COMFYUI_OUTPUT_DIR = self.original_output_dir
        server.jobs = self.original_jobs
        self.temp_dir.cleanup()

    def test_result_collection_uses_recorded_nested_output_metadata(self):
        video_bytes = b"video-bytes"
        flow_bytes = b"flow-png-bytes"
        server.jobs["job-1"] = {
            "status": "completed",
            "video_outputs": [{"filename": "wan.mp4", "subfolder": "worker_8188"}],
            "flow_images": [{"filename": "tracks.png", "subfolder": "worker_8188"}],
        }

        reader = AsyncMock(side_effect=[
            (video_bytes, "/output/worker_8188/wan.mp4"),
            (flow_bytes, "/output/worker_8188/tracks.png"),
        ])
        with patch.object(server, "_read_comfy_output", reader):
            result = asyncio.run(server._collect_job_results("job-1"))

        self.assertEqual(base64.b64decode(result["videos"][0]["data_base64"]), video_bytes)
        self.assertEqual(base64.b64decode(result["flow_images"][0]["data_base64"]), flow_bytes)
        self.assertEqual(
            [call.args for call in reader.await_args_list],
            [("wan.mp4", "worker_8188"), ("tracks.png", "worker_8188")],
        )

    def test_health_advertises_paired_wan_flow_contract(self):
        result = asyncio.run(server.health())

        self.assertGreaterEqual(result["contract_version"], 3)
        self.assertTrue(result["wan_full_inline_flow"])
        self.assertTrue(result["flow_only_available"])
        self.assertTrue(result["mode_aware_worker_affinity"])
        self.assertEqual(len(result["workers"]), result["worker_count"])

    def test_worker_pool_prefers_matching_warm_model_family(self):
        async def exercise_pool():
            pool = server.AffinityWorkerPool(["worker-0", "worker-1", "worker-2"])

            flow_worker = await pool.acquire("flow_only")
            await pool.release(flow_worker, "flow_only", succeeded=True)

            # A WAN job should use a cold worker before adding WAN to a worker
            # that has only loaded the flow stack.
            full_worker = await pool.acquire("full")
            await pool.release(full_worker, "full", succeeded=True)

            next_flow_worker = await pool.acquire("flow_only")
            await pool.release(next_flow_worker, "flow_only", succeeded=True)
            next_full_worker = await pool.acquire("full")
            await pool.release(next_full_worker, "full", succeeded=True)

            return flow_worker, full_worker, next_flow_worker, next_full_worker

        flow_worker, full_worker, next_flow_worker, next_full_worker = asyncio.run(
            exercise_pool()
        )
        self.assertEqual(flow_worker, "worker-0")
        self.assertEqual(full_worker, "worker-1")
        self.assertEqual(next_flow_worker, "worker-0")
        self.assertEqual(next_full_worker, "worker-1")

    def test_flow_nodes_do_not_pin_comfyui_generation_models(self):
        repo_root = Path(__file__).resolve().parents[2]
        sources = [
            repo_root / "remote_object_flow" / "comfyui" / "custom_nodes" / "novaplan" / "sam3" / "comfyui_node" / "nodes.py",
            repo_root / "remote_video_generation" / "custom_nodes" / "novaplan" / "cotracker3" / "nodes.py",
        ]
        for source_path in sources:
            source = source_path.read_text()
            with self.subTest(source=source_path):
                self.assertNotIn("current_loaded_models", source)
                self.assertNotIn("currently_used", source)
                self.assertNotIn("_PROTECTED_MODEL_REFS", source)
                self.assertIn("novaplan-model-ownership-v1", source)

    def test_video_only_custom_node_loader_supports_relative_imports(self):
        loader_path = (
            Path(__file__).resolve().parents[1]
            / "custom_nodes"
            / "novaplan"
            / "__init__.py"
        )
        source = loader_path.read_text()

        self.assertIn('parent_name = "novaplan_custom_nodes"', source)
        self.assertIn("submodule_search_locations=[str(init_path.parent)]", source)

    def test_server_launcher_preflights_required_worker_nodes(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "launch_server_video_generation.sh"
        ).read_text()
        for node_type in (
            "WanImageToVideo",
            "WanFirstLastFrameToVideo",
            "Sam3VideoNode",
            "CoTracker3Node",
        ):
            self.assertIn(node_type, launcher)
        self.assertIn("/object_info/", launcher)
        self.assertIn("novaplan-model-ownership-v1", launcher)
        self.assertIn("Stale or incompatible NovaPlan node implementations", launcher)

    def test_worker_launcher_replaces_role_workers_before_starting(self):
        repo_root = Path(__file__).resolve().parents[2]
        launcher = (
            repo_root / "remote_video_generation" / "launch_main_video_generation.sh"
        ).read_text()
        lifecycle = (repo_root / "scripts" / "comfyui_worker_lifecycle.sh").read_text()

        self.assertNotIn("KILL_OLD_PROCESSES", launcher)
        self.assertIn("stop_matching_processes", launcher)
        self.assertIn("stop_recorded_comfyui_workers", launcher)
        self.assertIn("video_worker_*.pid", launcher)
        self.assertIn("stop_comfyui_worker", launcher)
        self.assertIn("record_comfyui_worker_pid", launcher)
        self.assertIn("video_worker_${i}.pid", launcher)
        self.assertIn("pgrep -f", lifecycle)
        self.assertIn("ss -ltnp", lifecycle)
        self.assertIn("sync_video_runtime_custom_nodes", launcher)

    def test_video_workers_default_to_managed_vram(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "launch_main_video_generation.sh"
        ).read_text()

        self.assertIn('VIDEO_VRAM_MODE="${VIDEO_VRAM_MODE:-managed}"', launcher)
        self.assertIn('COMFYUI_EXTRA_ARGS="$COMFYUI_EXTRA_ARGS --highvram"', launcher)
        self.assertIn("highvram, managed", launcher)
        self.assertNotIn("--disable-smart-memory", launcher)

    def test_video_host_reuses_existing_comfyui_before_fresh_runtime_path(self):
        resolver = Path(__file__).resolve().parents[1] / "runtime_paths.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "comfyui"
            runtime = root / ".runtime" / "comfyui"
            legacy.mkdir(parents=True)
            runtime.mkdir(parents=True)
            (legacy / "main.py").write_text("# existing deployment\n")
            (runtime / "main.py").write_text("# newer empty deployment\n")

            env = os.environ.copy()
            env.pop("COMFYUI_DIR", None)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_video_comfyui_dir "$2"',
                    "resolver-test",
                    str(resolver),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(Path(result.stdout.strip()), legacy)

            explicit = root / "custom-comfyui"
            env["COMFYUI_DIR"] = str(explicit)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_video_comfyui_dir "$2"',
                    "resolver-test",
                    str(resolver),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(Path(result.stdout.strip()), explicit)

    def test_video_setup_refreshes_shared_flow_nodes(self):
        setup_source = (
            Path(__file__).resolve().parents[1] / "setup_video_generation_host.sh"
        ).read_text()

        self.assertIn('sync_video_runtime_custom_nodes "$REPO_ROOT" "$COMFYUI_DIR"', setup_source)

    def test_video_custom_node_sync_replaces_stale_runtime_without_clobbering_flow_assets(self):
        repo_root = Path(__file__).resolve().parents[2]
        resolver = repo_root / "remote_video_generation" / "runtime_paths.sh"
        video_nodes = repo_root / "remote_video_generation" / "custom_nodes" / "novaplan"
        flow_sam = repo_root / "remote_object_flow" / "comfyui" / "custom_nodes" / "novaplan" / "sam3"

        with tempfile.TemporaryDirectory() as directory:
            comfyui_dir = Path(directory) / "comfyui"
            target = comfyui_dir / "custom_nodes" / "novaplan"
            for relative in ("tapip3d_adapter", ".external/TAPIP3D", "moge2_metric_depth"):
                (target / relative).mkdir(parents=True)
            (target / "__init__.py").write_text("# existing full-flow loader\n")

            sam_target = target / "sam3" / "comfyui_node" / "nodes.py"
            cotracker_target = target / "cotracker3" / "nodes.py"
            sam_target.parent.mkdir(parents=True)
            cotracker_target.parent.mkdir(parents=True)
            sam_target.write_text("_PROTECTED_MODEL_REFS = []\n")
            cotracker_target.write_text("current_loaded_models = []\n")
            checkpoint = target / "cotracker3" / "scaled_offline.pth"
            checkpoint.write_bytes(b"checkpoint")
            unrelated = target / "moge2_metric_depth" / "keep.py"
            unrelated.write_text("# keep\n")

            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; sync_video_runtime_custom_nodes "$2" "$3"',
                    "sync-test",
                    str(resolver),
                    str(repo_root),
                    str(comfyui_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                sam_target.read_text(),
                (flow_sam / "comfyui_node" / "nodes.py").read_text(),
            )
            self.assertEqual(
                cotracker_target.read_text(),
                (video_nodes / "cotracker3" / "nodes.py").read_text(),
            )
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")
            self.assertEqual(unrelated.read_text(), "# keep\n")

    def test_full_workflow_contains_wan_and_inline_flow(self):
        request = server.JobRequest(
            mode="full",
            first_frame_base64="frame",
            mask_prompt="blue block",
            fps=12,
        )
        workflow = server.generate_unified_workflow("job-1", "start.png", request)
        classes = {node["class_type"] for node in workflow.values()}

        self.assertIn("WanImageToVideo", classes)
        self.assertIn("Sam3VideoNode", classes)
        self.assertIn("CoTracker3Node", classes)
        self.assertIn("SaveVideo", classes)
        self.assertIn("SaveImage", classes)
        self.assertIs(workflow["39"]["inputs"]["return_visualization"], False)
        self.assertEqual(workflow["14"]["inputs"]["fps"], 12)

    def test_generation_defaults_and_sampling_overrides_reach_workflow(self):
        default_request = server.JobRequest(mode="generate_only", first_frame_base64="frame")
        default_workflow = server.generate_unified_workflow("job-default", "start.png", default_request)

        self.assertEqual(default_request.frames, 41)
        self.assertEqual(default_request.fps, 16)
        self.assertEqual(default_workflow["8"]["inputs"]["length"], 41)
        self.assertEqual(default_workflow["14"]["inputs"]["fps"], 16)

        custom_request = server.JobRequest(
            mode="generate_only",
            first_frame_base64="frame",
            sampling_steps=8,
            guide_scale=2.25,
            sample_solver="unipc",
        )
        custom_workflow = server.generate_unified_workflow("job-custom", "start.png", custom_request)
        self.assertEqual(custom_workflow["11"]["inputs"]["steps"], 8)
        self.assertEqual(custom_workflow["11"]["inputs"]["end_at_step"], 4)
        self.assertEqual(custom_workflow["12"]["inputs"]["start_at_step"], 4)
        self.assertEqual(custom_workflow["11"]["inputs"]["cfg"], 2.25)
        self.assertEqual(custom_workflow["11"]["inputs"]["sampler_name"], "uni_pc")

    def test_flow_only_workflow_contains_no_wan_nodes(self):
        request = server.JobRequest(mode="flow_only", mask_prompt="blue block")
        workflow = server.generate_unified_workflow(
            "job-1",
            None,
            request,
            video_path="input.mp4",
        )
        classes = {node["class_type"] for node in workflow.values()}

        self.assertIn("Sam3VideoNode", classes)
        self.assertIn("CoTracker3Node", classes)
        self.assertTrue(classes.isdisjoint(server.WAN_WORKFLOW_CLASS_TYPES))
if __name__ == "__main__":
    unittest.main()
