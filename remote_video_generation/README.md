# Remote Video Generation

This folder contains the Wan 2.2 / ComfyUI video-generation host code:

- `setup_video_generation_host.sh`: one-time ComfyUI and model-folder setup.
- `launch_main_video_generation.sh`: starts ComfyUI worker processes.
- `launch_server_video_generation.sh`: starts the queue server on port `7000`.
- `video_generation_server.py`: FastAPI queue server.
- `tests/test_video_generation_server.py`: workstation connectivity test.

The video server can generate video candidates with WAN 2.2, run inline 2D
SAM3/CoTracker flow for candidate selection, and accept first-last-frame video generation requests for recovery videos.

## Install

On a fresh video GPU host, request access to the gated
[`facebook/sam3`](https://huggingface.co/facebook/sam3) weights, review the
Wan model terms, and then run from the repository root:

```bash
pixi run -e video-host hf auth login
DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host
```

`pixi run` creates or updates the `video-host` environment automatically. The setup command provisions ComfyUI, the video and inline 2D flow dependencies, and the default Wan model set.

The worker launcher refreshes the NovaPlan SAM3 and CoTracker adapters under
`$COMFYUI_DIR/custom_nodes` on every start. Rerun `setup-video-host` when
dependencies or model assets change.

The default SageAttention install may compile
CUDA code; select `ATTENTION_BACKEND=pytorch` when a compiler toolchain is not
available.

The scripts reuse an existing `<checkout>/comfyui` deployment first, including
its downloaded models. Fresh hosts use `<checkout>/.runtime/comfyui`. An
explicit `COMFYUI_DIR=/absolute/path/to/comfyui` has highest priority. Setup
pins fresh ComfyUI and SAM3 source checkouts to release revisions. Use
`COMFYUI_REF` or `SAM3_REF` only when deliberately selecting another tested
revision.

The default attention backend is SageAttention. On a fresh host, replace the setup command above with one of these alternatives when needed:

```bash
ATTENTION_BACKEND=flash DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host
ATTENTION_BACKEND=pytorch DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host
ATTENTION_BACKEND=none DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host
```

A successful fresh setup populates these model slots:

```text
$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
$COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
$COMFYUI_DIR/models/vae/wan_2.1_vae.safetensors
$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
```

## Launch

The reference deployment uses eight H100 GPUs with one ComfyUI worker per GPU.
This is an example chosen for parallel rollout throughput, not a minimum
hardware requirement. Smaller worker counts are supported, although they
reduce parallel throughput and non-H100 GPUs may require memory tuning.

Set `COMFYUI_NUM_WORKERS` to the number of GPUs visible to the launcher; the reference default is eight. Override host paths only when needed:

```bash
export COMFYUI_DIR=/absolute/path/to/comfyui
export COMFYUI_WORKER_START_PORT=8188
export COMFYUI_NUM_WORKERS=1  # Example for a single-GPU host
```

Input/output directories default under `COMFYUI_DIR`. The server derives the
exclusive worker end port from the start port and worker count and rejects an
inconsistent explicit `COMFYUI_WORKER_END_PORT`.

Launch in two separate terminals:

```bash
pixi run -e video-host launch-video-main
```

```bash
pixi run -e video-host launch-video-server
```

Inside `pixi shell -e video-host`, the scripts can also be run directly:

```bash
./remote_video_generation/launch_main_video_generation.sh
./remote_video_generation/launch_server_video_generation.sh
```

The server listens on `7000`; workers start at `8188`.
The worker launcher verifies the requested CUDA-device count and all Wan and
CoTracker assets before starting. It synchronizes the NovaPlan flow-node code
before replacing any running process. Before launching the queue API, the
server launcher checks every worker for the Wan I2V/FLF, SAM3, and CoTracker
node registrations and rejects incompatible flow nodes that violate the model
ownership contract. Set `REQUIRE_VIDEO_ASSETS=0` only for a deliberate
partial-runtime diagnostic.

The worker launcher always replaces its previous video workers. It tracks
their PIDs and falls back to matching the configured command lines and ports,
including workers that are still initializing. The queue-server launcher also
replaces its previous server by default.

### Kubernetes Port Forwarding

For a Kubernetes deployment, substitute the actual video-service pod and
namespace, then forward the queue API and first ComfyUI worker on the local workstation:

```bash
kubectl port-forward pod/<video-pod> -n <namespace> 7000:7000 8188:8188
export NOVAPLAN_VIDEO_SERVER_URL=http://127.0.0.1:7000
```

Port `7000` is the endpoint used by NovaPlan. Port `8188` is useful for direct
ComfyUI inspection; forwarding it is not required by the workstation client.
Pod names, namespaces, and any additional worker ports are deployment-specific.

Verify that the running process is the current paired-output server before
submitting a WAN batch:

```bash
curl http://127.0.0.1:7000/health
```

The response must include `"contract_version":3` and
`"wan_full_inline_flow":true`. The workstation client performs this inexpensive
check automatically so an incompatible server cannot consume a full WAN batch before
failing at candidate ranking.

### Optional Worker Warm-Up

After launching or restarting the remote workers, send one full-mode job to
each allocated GPU before starting a latency-sensitive closed-loop run. From
the workstation, set the public or forwarded queue URL and make
`--num_videos` match the remote `COMFYUI_NUM_WORKERS` value:

```bash
export NOVAPLAN_VIDEO_SERVER_URL=http://127.0.0.1:7000
pixi run -e local-planning-full warm-video-workers --num_videos 8
```

The reference command distributes eight independent `mode="full"` subjobs
from one batch across eight workers. Each worker therefore executes the
complete Wan plus inline SAM3/CoTracker3 pipeline once before closed-loop
execution. For a four-GPU deployment, use `--num_videos 4`; use the matching
value for other worker counts. The generated warm-up artifacts are stored under
`runs/verification/video_generation/`. This warm-up is optional and incurs one Wan generation per worker. 

## Verify

Forward the server if needed, then run its live installation check from the
workstation:

```bash
export NOVAPLAN_VIDEO_SERVER_URL=http://127.0.0.1:7000
pixi run -e local-planning-full verify-video-server
```

The check generates one Wan video, verifies that `full` mode returns both the
video and its inline CoTracker3 flow image, and then checks `flow_only` using
the generated video. The planner does not make the second request for Wan
candidates; it is the normal selection-flow path for external videos such as
Veo. Downloaded test artifacts are written under
`runs/verification/video_generation/`.

## Request Modes

`video_generation_server.py` supports:

- `mode="generate_only"`: WAN generation only.
- `mode="full"`: WAN generation plus inline 2D SAM3/CoTracker flow.
- `mode="flow_only"`: 2D flow on an already-generated video.

Veo candidates use `flow_only`, whose ComfyUI graph contains only video
loading, SAM3 segmentation, CoTracker3 inference, and flow-image saving. It
does not instantiate or execute any Wan node. SAM3 and CoTracker3 are cached
per persistent worker process, so each worker has one cold load and subsequent
jobs on that worker reuse the same model instances.

The queue server uses model-family affinity when assigning work. Wan `full`
and `generate_only` requests prefer workers that have already run Wan, while
Veo `flow_only` requests prefer workers that have already loaded the flow
stack. This avoids needlessly loading Wan onto a flow-only worker while still
allowing every idle worker to be used when the preferred workers are busy.
`GET /health` reports the queue's estimated residency for each worker.

For first-last-frame recovery generation, send `first_frame_base64`,
`last_frame_base64`, and `prompt` with `mode="generate_only"` or `mode="full"`.
Requests with `last_frame_base64` route through `WanFirstLastFrameToVideo`;
ordinary image-to-video requests keep using `WanImageToVideo`.

The explicit Wan client in [novaplan/video_generation/wan.py](../novaplan/video_generation/wan.py) wraps these details. Veo and hybrid orchestration live beside it in the same package.
