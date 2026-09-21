# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Focused, dependency-light tests for the exported depth workflow.

The ComfyUI runtime owns heavyweight modules such as torch and cv2.  These
tests execute the pure workflow/calibration helpers directly from their AST so
the release contract can still be checked in the lightweight development env.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from remote_object_flow.check_pointops_cuda import (
    assess_cuda_compatibility,
    parse_nvcc_cuda_version,
)
from remote_object_flow.ensure_cvd_runtime import REQUIRED_FILES, ensure_runtime
from remote_object_flow.ensure_raft_checkpoint import (
    MIN_CHECKPOINT_BYTES,
    ensure_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "remote_object_flow" / "object_flow_server.py"
NODE_BUNDLE_PATH = (
    ROOT / "remote_object_flow" / "comfyui" / "custom_nodes" / "novaplan"
)
NODE_PATH = (
    NODE_BUNDLE_PATH
    / "moge2_metric_depth"
    / "comfyui_node"
    / "node.py"
)
INIT_PATH = NODE_PATH.with_name("__init__.py")
CVD_PATH = NODE_PATH.with_name("cvd_optimizer.py")
TAPIP_INIT_PATH = (
    NODE_BUNDLE_PATH / "tapip3d_adapter" / "__init__.py"
)
TAPIP_NODES_PATH = TAPIP_INIT_PATH.with_name("nodes.py")
TAPIP_UPSTREAM_PATH = TAPIP_INIT_PATH.with_name("upstream.py")


def _load_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _workflow_builder():
    namespace = _load_functions(
        SERVER_PATH,
        {"get_depth_estimation_workflow"},
        {
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "SAVEVIDEO_FORMAT": "mp4",
            "SAVEVIDEO_CODEC": "h264",
            "_sanitize_job_id": lambda value: value.replace("-", "_"),
            "_log_debug": lambda *args, **kwargs: None,
            "_log_warning": lambda *args, **kwargs: None,
        },
    )
    return namespace["get_depth_estimation_workflow"]


def _flow_workflow_builder():
    namespace = _load_functions(
        SERVER_PATH,
        {"get_flow_extraction_workflow"},
        {
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "SAVEVIDEO_FORMAT": "mp4",
            "SAVEVIDEO_CODEC": "h264",
            "_sanitize_job_id": lambda value: value.replace("-", "_"),
            "_log_debug": lambda *args, **kwargs: None,
            "_log_warning": lambda *args, **kwargs: None,
        },
    )
    return namespace["get_flow_extraction_workflow"]


def _calibration_helpers():
    return _load_functions(
        NODE_PATH,
        {
            "_build_moge_infer_kwargs",
            "erode_mask",
            "build_calibration_mask",
            "stratified_subsample_by_depth",
            "estimate_scale_median_irls",
            "estimate_depth_scale_robust",
            "_fit_affine_ransac",
            "estimate_depth_affine_robust",
        },
        {"np": np, "inspect": inspect, "torch": object()},
    )


def test_workflow_uses_robust_moge_node_and_calibrated_cache_only():
    build = _workflow_builder()
    workflow = build(
        "job-id",
        "input.mp4",
        "first_depth.png",
        [600.0, 601.0, 320.0, 240.0],
    )["prompt"]

    depth = workflow["depth_estimation_job_id"]
    assert depth["class_type"] == "MoGe2CVDMetricDepthNode"
    assert "depth_backbone" not in depth["inputs"]
    assert depth["inputs"]["calibration_mode"] == "Affine (s * d + t)"
    assert depth["inputs"]["enable_cvd"] == "enabled"
    assert depth["inputs"]["ransac_iters"] == 1000
    assert depth["inputs"]["inlier_threshold"] == 0.15
    assert depth["inputs"]["min_region_size"] == 50
    assert workflow["cache_raw_depth_job_id"]["inputs"]["raw_depth"] == [
        "depth_estimation_job_id",
        2,
    ]

    node_tree = ast.parse(NODE_PATH.read_text())
    node_class = next(
        node
        for node in node_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoGe2CVDMetricDepthNode"
    )
    return_names = next(
        node.value
        for node in node_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RETURN_NAMES" for target in node.targets)
    )
    assert ast.literal_eval(return_names) == (
        "input_depth_visual",
        "depth_video",
        "calibrated_depth",
    )
    assert "uncalibrated" not in NODE_PATH.read_text().lower()
    assert "return_uncalibrated" not in SERVER_PATH.read_text()
    assert "MoGe2CVDMetricDepthNode" in NODE_PATH.read_text()

    node_source = NODE_PATH.read_text()
    assert "freeze_shift=True" in node_source
    assert "depth_backbone" not in node_source
    assert '"Scale Only (s)"' not in node_source
    assert node_source.index("# ====== CVD Optimization ======") < node_source.index("# Calibration")
    assert "depth_for_vis = depth_for_vis * scale + shift" in node_source
    assert "depth_resized = depth_resized * scale + shift" in node_source

    server_tree = ast.parse(SERVER_PATH.read_text())
    request_class = next(
        node
        for node in server_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DepthEstimationRequest"
    )
    request_fields = {
        node.target.id: node
        for node in request_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    cvd_default = next(
        keyword.value
        for keyword in request_fields["enable_cvd"].value.keywords
        if keyword.arg == "default"
    )
    depth_min_default = next(
        keyword.value
        for keyword in request_fields["valid_depth_min"].value.keywords
        if keyword.arg == "default"
    )
    assert ast.literal_eval(cvd_default) is True
    assert ast.literal_eval(depth_min_default) == 0.6


def test_shared_host_setup_preserves_and_validates_flow_node_bundle():
    repo_root = Path(__file__).resolve().parents[2]
    pixi_manifest = (repo_root / "pixi.toml").read_text()
    video_setup = (repo_root / "remote_video_generation" / "setup_video_generation_host.sh").read_text()
    video_runtime_paths = (repo_root / "remote_video_generation" / "runtime_paths.sh").read_text()
    flow_setup = (repo_root / "remote_object_flow" / "setup_object_flow_host.sh").read_text()
    flow_launcher = (repo_root / "remote_object_flow" / "launch_server_object_flow.sh").read_text()
    flow_worker_launcher = (repo_root / "remote_object_flow" / "launch_main_object_flow.sh").read_text()

    assert 'setuptools = ">=68,<82"' in pixi_manifest
    assert 'source "$SCRIPT_DIR/runtime_paths.sh"' in video_setup
    assert 'sync_video_runtime_custom_nodes "$REPO_ROOT" "$COMFYUI_DIR"' in video_setup
    assert '-d "$target/tapip3d_adapter"' in video_runtime_paths
    assert '-d "$target/.external/TAPIP3D"' in video_runtime_paths
    assert '-d "$target/moge2_metric_depth"' in video_runtime_paths
    assert 'sam_source="$flow_sam_source"' in video_runtime_paths
    assert 'cp -a "$video_source/cotracker3/nodes.py" "$target/cotracker3/nodes.py"' in video_runtime_paths
    assert "MoGe2CVDMetricDepthNode" in flow_launcher
    assert "/object_info/" in flow_launcher
    assert "setup-object-flow-host" in flow_launcher
    assert "Migrated the discovered RAFT checkpoint" in flow_setup
    assert "moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth" in flow_setup
    assert "RAFT_CHECKPOINT_SOURCE" in flow_setup
    assert "RAFT_MODELS_URL" in flow_setup
    assert 'DOWNLOAD_OBJECT_FLOW_MODELS="${DOWNLOAD_OBJECT_FLOW_MODELS:-1}"' in flow_setup
    assert "INSTALL_CVD_RUNTIME" in flow_setup
    assert "INSTALL_CVD_RUNTIME" in flow_launcher
    assert "ensure_cvd_runtime.py" in flow_setup
    assert "ensure_cvd_runtime.py" in flow_launcher
    assert "ensure_raft_checkpoint.py" in flow_setup
    assert "ensure_raft_checkpoint.py" in flow_launcher
    assert "INSTALL_TAPIP3D_POINTOPS" in flow_setup
    assert "INSTALL_TAPIP3D_SOURCE" in flow_setup
    assert "TAPIP3D_REPO" in flow_setup
    assert 'TAPIP3D_REF="${TAPIP3D_REF:-4cb7e69a' in flow_setup
    assert "clone_tapip3d" in flow_setup
    assert "NOVAPLAN_TAPIP3D_DIR" in flow_setup
    assert "pointops2" in flow_setup
    assert "check_pointops_cuda.py" in flow_setup
    assert "verify_tapip3d_adapter.py" in flow_setup
    assert "--require-pointops-cuda" in flow_setup
    assert '"numpy>=1.26,<2"' in flow_setup
    assert '"opencv-python>=4.8,<5"' in flow_setup
    assert '"plyfile>=1.0,<1.1.4"' in flow_setup
    assert "pip check" in flow_setup
    assert "REQUIRE_OBJECT_FLOW_ASSETS" in flow_worker_launcher
    assert "tapip3d_final.pth" in flow_worker_launcher


def test_flow_worker_launcher_replaces_role_workers_before_starting():
    repo_root = Path(__file__).resolve().parents[2]
    launcher = (repo_root / "remote_object_flow" / "launch_main_object_flow.sh").read_text()
    lifecycle = (repo_root / "scripts" / "comfyui_worker_lifecycle.sh").read_text()

    assert "KILL_OLD_PROCESSES" not in launcher
    assert "stop_matching_processes" in launcher
    assert "stop_recorded_comfyui_workers" in launcher
    assert "flow_worker_*.pid" in launcher
    assert "stop_comfyui_worker" in launcher
    assert "record_comfyui_worker_pid" in launcher
    assert "flow_worker_${i}.pid" in launcher
    assert "pgrep -f" in lifecycle
    assert "ss -ltnp" in lifecycle


def test_cvd_runtime_provisioning_reuses_minimal_source_tree():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "existing" / "cvd_opt"
        destination = root / "installed" / "cvd_opt"
        for relative in REQUIRED_FILES:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"# {relative}\n"
            if relative == Path("core/raft.py"):
                content += (
                    "from corr import CorrBlock\n"
                    "from utils.utils import coords_grid\n"
                    "# return coords1 - coords0, flow_up, net\n"
                )
            elif relative == Path("core/corr.py"):
                content += "from utils.utils import bilinear_sampler\n"
            path.write_text(content)

        result = ensure_runtime(
            destination,
            source=None,
            search_root=root,
            source_base_url="unused://source",
        )

        assert all((destination / relative).is_file() for relative in REQUIRED_FILES)
        assert "from .corr import CorrBlock" in (destination / "core/raft.py").read_text()
        assert "from .utils.utils import coords_grid" in (destination / "core/raft.py").read_text()
        assert "from .utils.utils import bilinear_sampler" in (destination / "core/corr.py").read_text()
        assert "Modified by Robotics and AI Institute LLC" in (
            destination / "core/raft.py"
        ).read_text()
        assert "Apache License" in (destination / "LICENSE").read_text()
        assert "BSD 3-Clause License" in (destination / "RAFT_LICENSE").read_text()
        assert result.startswith("reused ")


def test_raft_checkpoint_provisioning_reuses_existing_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        existing = root / "old" / "raft-things.pth"
        destination = root / "new" / "raft-things.pth"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"r" * MIN_CHECKPOINT_BYTES)

        result = ensure_checkpoint(
            destination,
            source=None,
            search_root=root,
            archive_url="unused://archive",
        )

        assert destination.read_bytes() == existing.read_bytes()
        assert result.startswith("reused ")


def test_raft_checkpoint_provisioning_extracts_model_archive():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "models.zip"
        destination = root / "checkpoints" / "raft-things.pth"
        payload = b"r" * MIN_CHECKPOINT_BYTES
        with zipfile.ZipFile(archive, "w") as models:
            models.writestr("models/raft-things.pth", payload)

        result = ensure_checkpoint(
            destination,
            source=None,
            search_root=None,
            archive_url=archive.as_uri(),
        )

        assert destination.read_bytes() == payload
        assert result == f"downloaded to {destination}"


def test_exported_node_bundle_has_one_named_depth_backend():
    bundle_init = (NODE_BUNDLE_PATH / "__init__.py").read_text()
    server_source = SERVER_PATH.read_text()

    assert '"moge2_metric_depth"' in bundle_init
    assert 'depth_node_class = "MoGe2CVDMetricDepthNode"' in server_source
    assert '"depth_backbone"' not in server_source


def test_cvd_uses_native_camera_transforms_without_lietorch():
    source = CVD_PATH.read_text()

    assert "from lietorch" not in source
    assert "torch.linalg.inv(" in source
    assert "camera_matrices_opt = camera_matrices.clone()" in source
    assert 'NOVAPLAN_NODE_ROOT / ".external" / "mega-sam" / "cvd_opt"' in source
    assert "class _RAFTArgs(Namespace)" in source
    assert "def __contains__" in source
    assert 'RAFT_PACKAGE_NAME = "_novaplan_cvd_raft"' in source
    assert "submodule_search_locations=[str(RAFT_CORE_DIR)]" in source
    assert "sys.path.insert" not in source
    assert "del sys.modules" not in source


def test_cvd_can_be_explicitly_disabled():
    build = _workflow_builder()
    workflow = build(
        "cvd-off",
        "input.mp4",
        "first_depth.png",
        [600.0, 600.0, 320.0, 240.0],
        depth_model="moge2",
        enable_cvd=False,
    )["prompt"]
    assert workflow["depth_estimation_cvd_off"]["inputs"]["enable_cvd"] == "disabled"


def test_ransac_recovers_first_frame_affine_with_outliers():
    helpers = _calibration_helpers()
    estimate = helpers["estimate_depth_affine_robust"]
    rng = np.random.default_rng(7)
    predicted = rng.uniform(0.75, 2.5, size=(48, 48)).astype(np.float32)
    metric = 1.65 * predicted + 0.23
    outliers = rng.random(predicted.shape) < 0.2
    metric[outliers] = rng.uniform(0.8, 5.5, size=int(outliers.sum()))

    scale, shift, success = estimate(
        predicted,
        metric,
        valid_min=0.7,
        valid_max=6.0,
        method="ransac",
        num_ransac_iters=1000,
        inlier_threshold=0.05,
        filter_pepper=False,
        use_planar=False,
        use_ratio_consistency=False,
        erosion_px=0,
        min_region_size=50,
        min_pixels=500,
        seed=42,
    )

    assert success
    assert abs(scale - 1.65) <= 0.03
    assert abs(shift - 0.23) <= 0.04


def test_ransac_uses_paper_absolute_metric_residual_threshold():
    source = NODE_PATH.read_text()

    assert "inlier_mask = diff < inlier_threshold" in source
    assert "inliers = final_residuals < inlier_threshold" in source
    assert "np.maximum(gt_valid, 0.1) * inlier_threshold" not in source
    assert "Absolute RANSAC residual threshold in meters" in source


def test_modular_flow_test_explicitly_enables_paper_cvd_defaults():
    test_source = (ROOT / "remote_object_flow" / "tests" / "test_object_flow_server.py").read_text()

    assert '"enable_cvd": not args.disable_cvd' in test_source
    assert '"cvd_w_grad": args.cvd_w_grad' in test_source
    assert '"cvd_w_normal": args.cvd_w_normal' in test_source
    assert '"cvd_resolution": args.cvd_resolution' in test_source
    assert 'default=196608' in test_source


def test_closed_loop_cli_can_disable_optional_cvd_end_to_end():
    entrypoint = (ROOT / "local_planning" / "run_closed_loop_execution.py").read_text()
    compiler = (ROOT / "local_planning" / "run_execution_step.py").read_text()
    orchestrator = (ROOT / "novaplan" / "closed_loop_execution.py").read_text()

    assert '"--disable_cvd"' in entrypoint
    assert "enable_cvd=not args.disable_cvd" in entrypoint
    assert "if not config.enable_cvd:" in orchestrator
    assert 'cmd.append("--disable_cvd")' in orchestrator
    assert "if args.disable_cvd:" in compiler
    assert 'cmd.append("--disable_cvd")' in compiler


def test_affine_requires_configured_minimum_pixels():
    helpers = _calibration_helpers()
    estimate = helpers["estimate_depth_affine_robust"]
    predicted = np.ones((10, 10), dtype=np.float32)
    metric = predicted * 1.5 + 0.2
    assert estimate(
        predicted,
        metric,
        filter_pepper=False,
        erosion_px=0,
        min_pixels=101,
    ) == (1.0, 0.0, False)


def test_failed_affine_cannot_be_reported_as_successful_scale_fallback():
    estimate = _calibration_helpers()["estimate_depth_affine_robust"]
    rng = np.random.default_rng(9)
    predicted = rng.uniform(0.7, 2.5, size=(48, 48)).astype(np.float32)
    unrelated_metric = rng.uniform(0.7, 5.5, size=(48, 48)).astype(np.float32)

    _, shift, success = estimate(
        predicted,
        unrelated_metric,
        valid_min=0.6,
        valid_max=6.0,
        method="ransac",
        num_ransac_iters=1000,
        inlier_threshold=0.01,
        filter_pepper=False,
        use_planar=False,
        use_ratio_consistency=False,
        erosion_px=0,
        min_region_size=50,
        min_pixels=500,
    )

    assert shift == 0.0
    assert success is False


def test_depth_history_requires_success_and_cache_acknowledgement():
    validator = _load_functions(
        SERVER_PATH,
        {"_validate_depth_history"},
        {"Any": Any, "Dict": Dict},
    )["_validate_depth_history"]
    valid = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {"cache_node": {"cached": ["depth_job"]}},
    }
    assert validator(valid, "cache_node", "depth_job") is valid["outputs"]

    invalid_histories = (
        {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_error", {"exception_message": "calibration failed"}]
                ],
            },
            "outputs": {},
        },
        {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {},
        },
    )
    for history in invalid_histories:
        try:
            validator(history, "cache_node", "depth_job")
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid Comfy depth history was accepted")


def test_cached_depth_flow_keeps_originating_worker_affinity():
    least_loaded_calls = []

    async def least_loaded():
        least_loaded_calls.append(True)
        return "http://worker-new:8187"

    select = _load_functions(
        SERVER_PATH,
        {"_select_flow_worker"},
        {
            "Optional": Optional,
            "_get_least_loaded_worker": least_loaded,
        },
    )["_select_flow_worker"]

    assert (
        asyncio.run(select("cached", "http://worker-origin:8187"))
        == "http://worker-origin:8187"
    )
    assert least_loaded_calls == []
    assert asyncio.run(select("npy", None)) == "http://worker-new:8187"
    assert least_loaded_calls == [True]

    server_source = SERVER_PATH.read_text()
    assert '"worker_url": target_url' in server_source
    assert "target_url = await _select_flow_worker(depth_mode, cached_worker_url)" in server_source


def test_ransac_never_returns_an_out_of_bounds_shift():
    fit = _calibration_helpers()["_fit_affine_ransac"]
    predicted = np.linspace(0.7, 2.5, 1000)
    # Every exact two-point model has shift=4, outside the configured bounds.
    metric = 1.2 * predicted + 4.0
    _, shift, _ = fit(
        predicted,
        metric,
        num_iterations=1000,
        shift_bounds=(-3.0, 3.0),
    )
    assert -3.0 <= shift <= 3.0


def test_known_intrinsics_are_forwarded_as_moge_fov():
    helper = _calibration_helpers()["_build_moge_infer_kwargs"]

    def infer(image, fov_x=None, apply_mask=True):
        del image, fov_x, apply_mask

    K = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    kwargs = helper(infer, K, height=480, width=640, device=None)
    expected_fov = np.degrees(2.0 * np.arctan(640.0 / (2.0 * 500.0)))
    assert abs(kwargs["fov_x"] - expected_fov) <= 1e-9
    assert kwargs["apply_mask"] is False


def test_execution_flow_graph_uses_sam3_and_vanilla_tapip3d():
    build = _flow_workflow_builder()
    workflow = build(
        "flow-job",
        "selected.mp4",
        "depth_flow_job",
        [600.0, 601.0, 320.0, 240.0],
        "blue block",
        "checkpoints/tapip3d_final.pth",
        2,
        32,
        0.9,
        tapip3d_node="sliding",  # compatibility value must not change deployment
    )["prompt"]
    sam = workflow["sam3_flow_job"]
    tapip = workflow["tapip3d_flow_job"]
    assert sam["class_type"] == "Sam3VideoNode"
    assert sam["inputs"]["prompt"] == "blue block"
    assert tapip["class_type"] == "TAPIP3DNode"
    assert tapip["inputs"]["mask"] == ["sam3_flow_job", 1]
    assert tapip["inputs"]["depth"] == ["load_cached_depth_flow_job", 0]
    assert tapip["inputs"]["intrinsics"] == ["intrinsics_flow_job", 0]
    assert "from .nodes import NODE_CLASS_MAPPINGS" in TAPIP_INIT_PATH.read_text()
    assert '"TAPIP3DNode": TAPIP3DNode' in TAPIP_NODES_PATH.read_text()


def test_tapip3d_adapter_uses_unmodified_pinned_upstream_checkout():
    node_source = TAPIP_NODES_PATH.read_text()
    upstream_source = TAPIP_UPSTREAM_PATH.read_text()

    assert "from .upstream import" in node_source
    assert 'importlib.import_module("utils.inference_utils")' in upstream_source
    assert "TAPIP3D_ROOT" in upstream_source
    assert "_POINTOPS.knnquery = _knnquery_torch" in upstream_source
    assert "with _prepared_pointops_import()" in upstream_source
    assert 'types.ModuleType("pointops2")' in upstream_source
    assert 'types.ModuleType("pointops2_cuda")' in upstream_source
    assert "_isolated_upstream_imports" in upstream_source
    assert "novaplan_tapip3d_upstream" in upstream_source
    assert "sys.modules.update(preserved)" in upstream_source
    assert "tap_models" not in node_source
    assert "tap_utils" not in node_source
    assert not (NODE_BUNDLE_PATH / "TAPIP3D").exists()
    assert not TAPIP_INIT_PATH.with_name("node_adaptive.py").exists()
    assert not TAPIP_INIT_PATH.with_name("node_new.py").exists()

    bundle_source = (NODE_BUNDLE_PATH / "__init__.py").read_text()
    assert bundle_source.index('"tapip3d":') < bundle_source.index(
        '"moge2_metric_depth":'
    )


def test_pointops_cuda_compatibility_matches_pytorch_policy():
    assert (
        parse_nvcc_cuda_version(
            "Cuda compilation tools, release 12.1, V12.1.105"
        )
        == "12.1"
    )

    compatible, detail = assess_cuda_compatibility(
        "12.8",
        "12.1",
        Path("/usr/local/cuda-12.1/bin/nvcc"),
    )
    assert compatible
    assert "minor-version difference" in detail

    compatible, detail = assess_cuda_compatibility(
        "13.0",
        "12.1",
        Path("/usr/local/cuda-12.1/bin/nvcc"),
    )
    assert not compatible
    assert "incompatible" in detail

    compatible, detail = assess_cuda_compatibility(
        None,
        "12.1",
        Path("/usr/local/cuda-12.1/bin/nvcc"),
    )
    assert not compatible
    assert "does not include CUDA support" in detail


def test_tapip3d_adapter_imports_without_pointops():
    with tempfile.TemporaryDirectory() as directory:
        tapip3d_root = Path(directory) / "TAPIP3D"

        files = {
            "LICENSE": "test fixture\n",
            "models/__init__.py": (
                "from third_party.pointops2.functions import pointops\n"
            ),
            "utils/__init__.py": "",
            "utils/inference_utils.py": textwrap.dedent(
                """
                import models

                def load_model(*args, **kwargs):
                    return None

                def inference(*args, **kwargs):
                    return None

                def get_grid_queries(*args, **kwargs):
                    return None

                def resize_depth_bilinear(*args, **kwargs):
                    return None
                """
            ),
            "third_party/__init__.py": "",
            "third_party/pointops2/__init__.py": "",
            "third_party/pointops2/functions/__init__.py": "from pointops2 import *\n",
            "third_party/pointops2/functions/pointops.py": textwrap.dedent(
                """
                import pointops2_cuda as pointops_cuda

                def knnquery(*args, **kwargs):
                    return "cuda"
                """
            ),
        }
        for relative, content in files.items():
            path = tapip3d_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        probe = textwrap.dedent(
            """
            import importlib.abc
            import importlib.util
            import os
            import sys
            import types

            class MissingPointops(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in {"pointops2", "pointops2_cuda"}:
                        raise ModuleNotFoundError(fullname)
                    return None

            sys.meta_path.insert(0, MissingPointops())
            sys.modules["torch"] = types.ModuleType("torch")
            os.environ["NOVAPLAN_TAPIP3D_DIR"] = sys.argv[2]

            spec = importlib.util.spec_from_file_location(
                "_tapip3d_fallback_probe",
                sys.argv[1],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load adapter probe")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            assert module._HAS_POINTOPS_CUDA is False
            assert module._POINTOPS.knnquery is module._knnquery_torch
            assert "pointops2" not in sys.modules
            assert "pointops2_cuda" not in sys.modules
            """
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                probe,
                str(TAPIP_UPSTREAM_PATH),
                str(tapip3d_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"adapter fallback probe failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


class DepthCalibrationContractTest(unittest.TestCase):
    test_workflow_uses_robust_moge_node_and_calibrated_cache_only = staticmethod(
        test_workflow_uses_robust_moge_node_and_calibrated_cache_only
    )
    test_cvd_can_be_explicitly_disabled = staticmethod(
        test_cvd_can_be_explicitly_disabled
    )
    test_ransac_recovers_first_frame_affine_with_outliers = staticmethod(
        test_ransac_recovers_first_frame_affine_with_outliers
    )
    test_ransac_uses_paper_absolute_metric_residual_threshold = staticmethod(
        test_ransac_uses_paper_absolute_metric_residual_threshold
    )
    test_modular_flow_test_explicitly_enables_paper_cvd_defaults = staticmethod(
        test_modular_flow_test_explicitly_enables_paper_cvd_defaults
    )
    test_closed_loop_cli_can_disable_optional_cvd_end_to_end = staticmethod(
        test_closed_loop_cli_can_disable_optional_cvd_end_to_end
    )
    test_shared_host_setup_preserves_and_validates_flow_node_bundle = staticmethod(
        test_shared_host_setup_preserves_and_validates_flow_node_bundle
    )
    test_cvd_runtime_provisioning_reuses_minimal_source_tree = staticmethod(
        test_cvd_runtime_provisioning_reuses_minimal_source_tree
    )
    test_raft_checkpoint_provisioning_reuses_existing_file = staticmethod(
        test_raft_checkpoint_provisioning_reuses_existing_file
    )
    test_raft_checkpoint_provisioning_extracts_model_archive = staticmethod(
        test_raft_checkpoint_provisioning_extracts_model_archive
    )
    test_exported_node_bundle_has_one_named_depth_backend = staticmethod(
        test_exported_node_bundle_has_one_named_depth_backend
    )
    test_cvd_uses_native_camera_transforms_without_lietorch = staticmethod(
        test_cvd_uses_native_camera_transforms_without_lietorch
    )
    test_affine_requires_configured_minimum_pixels = staticmethod(
        test_affine_requires_configured_minimum_pixels
    )
    test_failed_affine_cannot_be_reported_as_successful_scale_fallback = staticmethod(
        test_failed_affine_cannot_be_reported_as_successful_scale_fallback
    )
    test_depth_history_requires_success_and_cache_acknowledgement = staticmethod(
        test_depth_history_requires_success_and_cache_acknowledgement
    )
    test_cached_depth_flow_keeps_originating_worker_affinity = staticmethod(
        test_cached_depth_flow_keeps_originating_worker_affinity
    )
    test_ransac_never_returns_an_out_of_bounds_shift = staticmethod(
        test_ransac_never_returns_an_out_of_bounds_shift
    )
    test_known_intrinsics_are_forwarded_as_moge_fov = staticmethod(
        test_known_intrinsics_are_forwarded_as_moge_fov
    )
    test_execution_flow_graph_uses_sam3_and_vanilla_tapip3d = staticmethod(
        test_execution_flow_graph_uses_sam3_and_vanilla_tapip3d
    )
    test_tapip3d_adapter_uses_unmodified_pinned_upstream_checkout = staticmethod(
        test_tapip3d_adapter_uses_unmodified_pinned_upstream_checkout
    )
    test_pointops_cuda_compatibility_matches_pytorch_policy = staticmethod(
        test_pointops_cuda_compatibility_matches_pytorch_policy
    )
    test_tapip3d_adapter_imports_without_pointops = staticmethod(
        test_tapip3d_adapter_imports_without_pointops
    )


if __name__ == "__main__":
    unittest.main()
