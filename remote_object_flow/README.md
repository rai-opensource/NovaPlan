# Remote Object Flow

This folder deploys NovaPlan's metric 3D object flow grounding module. It combines
MoGe2 depth, optional CVD temporal refinement, robust metric calibration, SAM3
segmentation, and TAPIP3D tracking. The resulting object flow is used to compute
execution transforms.

## Depth and Flow Contract

```text
RGB rollout + aligned initial sensor depth + camera intrinsics
  -> MoGe2 sequence depth
  -> optional CVD refinement
  -> robust frame-0 affine calibration
  -> calibrated metric depth for the full video
  -> SAM3 object mask + TAPIP3D tracks
```

The initial depth image must be aligned with the RGB image. After optional
CVD, NovaPlan fits `D_metric = s * D_prediction + t` on frame 0 and applies the
same parameters to every frame. Downstream tracking uses only calibrated
numeric depth. Optional depth videos support visual inspection of the estimate.

CVD uses the pinned RAFT source subset and checkpoint provisioned by setup. It
does not run the rest of the upstream camera/depth/reconstruction pipeline.
Disable it per request with `enable_cvd=false` or from the closed-loop CLI with
`--disable_cvd`.

## Install

On a fresh object-flow GPU host, request access to the gated
[`facebook/sam3`](https://huggingface.co/facebook/sam3) weights, then run from
the repository root:

```bash
pixi run -e object-flow-host hf auth login
pixi run -e object-flow-host setup-object-flow-host
```

`pixi run` installs or updates the `object-flow-host` environment automatically
before running setup, so a separate `pixi install` command is not needed.

Set an absolute `COMFYUI_DIR` before setup and launch to reuse another
deployment. Setup pins ComfyUI, SAM3, MoGe, and the original TAPIP3D repository
to release revisions, installs their inference-time dependencies, and
provisions these assets:

```text
$COMFYUI_DIR/custom_nodes/novaplan/
  tapip3d_adapter/
  .external/TAPIP3D/
    LICENSE
    utils/inference_utils.py
    checkpoints/tapip3d_final.pth
  moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth
  .external/mega-sam/cvd_opt/
    LICENSE
    RAFT_LICENSE
    core/raft.py
```

The official TAPIP3D files remain unmodified. NovaPlan's ComfyUI bridge and
optional torch-only KNN fallback live in `tapip3d_adapter/`; the faster
upstream `pointops2` extension is built automatically when a compatible
`nvcc` is available. The full upstream monocular demo stack is unnecessary
because NovaPlan supplies calibrated depth and camera parameters.

The setup command downloads the pinned TAPIP3D source and its public checkpoint
by default. SAM3 and MoGe2 weights populate the authenticated Hugging Face cache
on the first model call.

Reuse local CVD assets on an offline host with:

```bash
RAFT_CHECKPOINT_SOURCE=/path/to/raft-things.pth \
CVD_RUNTIME_SOURCE=/path/to/cvd_opt \
  pixi run -e object-flow-host setup-object-flow-host
```

Control the optional CUDA extension with
`INSTALL_TAPIP3D_POINTOPS=auto|1|0`. Use `INSTALL_CVD_RUNTIME=0` only when every
request will disable CVD.

For an audited mirror or offline checkout, override `SAM3_REF`, `MOGE_REF`,
`TAPIP3D_REPO`, `TAPIP3D_REF`, or `NOVAPLAN_TAPIP3D_DIR`. Set
`INSTALL_TAPIP3D_SOURCE=0` only when `NOVAPLAN_TAPIP3D_DIR` already contains
the official source and license.

## Launch

The reference deployment uses one H100 GPU with one ComfyUI worker. This is an
example, not a minimum hardware requirement; compatible CUDA GPUs may be used
with suitable memory settings. Override defaults only when needed:

```bash
export COMFYUI_DIR=/absolute/path/to/comfyui
export COMFYUI_START_PORT=8187
export COMFYUI_NUM_WORKERS=1
```

Launch the worker and queue API in separate terminals:

```bash
pixi run -e object-flow-host launch-object-flow-main
```

```bash
pixi run -e object-flow-host launch-object-flow-server
```

The API defaults to port `7001`; the worker starts at `8187`. Each main launcher always replaces prior NovaPlan object-flow workers, including workers still
initializing. The server launcher replaces its prior queue server by default.

For a Kubernetes deployment, substitute the actual pod and namespace, and run the commands below on the local workstation:

```bash
kubectl port-forward pod/<object-flow-pod> -n <namespace> 7001:7001 8187:8187
export NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://127.0.0.1:7001
```

NovaPlan uses port `7001`; port `8187` is only needed for direct ComfyUI
inspection.

The primary endpoints are:

```text
GET  /health
POST /jobs/depth_estimation
POST /jobs/flow_extraction
GET  /status/{job_id}
GET  /result/{job_id}/flow
GET  /result/{job_id}/coords_3d
GET  /result/{job_id}/visibilities
```

Depth arrays are cached in the worker that produced them. A subsequent flow job
using `depth_source_job_id` is routed back to that worker; caches do not survive
a worker restart.

## Verify

After forwarding port `7001`, run the live installation check from the
workstation:

```bash
export NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://127.0.0.1:7001
pixi run -e local-planning-full verify-object-flow-server
```

The check submits the recorded first rollout, runs MoGe2/CVD calibration, SAM3
segmentation, and TAPIP3D tracking, and validates the returned metric object
flow. Outputs are written under `runs/verification/object_flow/`.

### Debug media

Request the complete flow-debug bundle from the workstation with:

```bash
pixi run -e local-planning-full python \
  remote_object_flow/tests/test_object_flow_server.py \
  --server "$NOVAPLAN_OBJECT_FLOW_SERVER_URL" \
  --example_dir example_data/color_sorting \
  --step step_001 \
  --mask_prompt "blue cube" \
  --debug_artifacts \
  --debug_output_dir runs/verification/object_flow/debug_artifacts
```

`--debug_artifacts` requests the colorized depth-estimation video, SAM3
segmentation video, TAPIP3D tracking video, static flow image, and auxiliary
arrays. To request only selected media, replace it with any combination of
`--depth_debug_video`, `--sam3_debug_video`, and
`--tapip3d_debug_video`. Add `--auxiliary_arrays` when the numeric masks,
depths, intrinsics, extrinsics, and query points are also needed.

During closed-loop execution, add `--flow_debug_artifacts` to request the same
complete bundle for each selected normal or recovery rollout. The lighter
`--flow_sam3_debug_video` option requests only the SAM3 debug video. Closed-loop
debug outputs are grouped beneath the corresponding
`runs/closed_loop/<timestamp>/step_XXX/debug_artifacts/` directory.

## Troubleshooting

- A missing-node error means the worker is running an older custom-node copy.
  Rerun `setup-object-flow-host`, then restart both launch commands.
- For a missing TAPIP3D or RAFT checkpoint, rerun setup with
  `DOWNLOAD_OBJECT_FLOW_MODELS=1`, or by supplying the documented source path.
- For output file names and transform conventions, use the
  [local planning output contract](../local_planning/README.md#outputs-and-review)
  and the test client's `--help` output.
- SAM3, TAPIP3D, MoGe2, MegaSAM CVD, RAFT, and their model assets retain their
  upstream terms and are not covered by NovaPlan's MIT License.
