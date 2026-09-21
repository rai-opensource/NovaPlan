# Local Planning and Closed-Loop Execution

This folder is the workstation entry point for NovaPlan. It covers task-structure
assessment, strategic or reactive action selection, video rollout ranking,
geometric grounding, state verification, recovery, and optional Viser visualization.
Given an initial RGB-D observation and a high-level task command, a VLM planner
proposes and verifies candidate video rollouts. NovaPlan adaptively grounds the
selected future video with object-centric or human hand flow, converts that motion
into camera-frame relative end-effector transforms, and uses new observations
to verify each outcome and generate local recovery actions when needed.

Run commands below from the repository root.

## Install

Install the full workstation environment:

```bash
pixi install -e local-planning-full
```

This includes the core planner plus the `hand` dependencies (`open3d`,
`trimesh`, and `pyrender`) and `viz` dependencies (`viser` and `trimesh`). Hand
flow calibration is part of normal closed-loop execution because any step may
switch from unreliable object flow to hand flow.

The service endpoints default to local forwarded ports:

```text
NOVAPLAN_VIDEO_SERVER_URL=http://127.0.0.1:7000
NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://127.0.0.1:7001
NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict
```

Set `OPENAI_API_KEY` for VLM calls. NovaPlan defaults all OpenAI VLM tasks to
`gpt-5.6`. The resolved model is printed when `VLMAdapter` starts. Override the shared default with
`NOVAPLAN_VLM_MODEL` (or `OPENAI_MODEL`), or set task-specific variables such
as `NOVAPLAN_HORIZON_VLM_MODEL`, `NOVAPLAN_VERIFICATION_VLM_MODEL`, and
`NOVAPLAN_RECOVERY_VLM_MODEL`.

Runs using Veo also require `GOOGLE_CLOUD_PROJECT` and valid Google Application
Default Credentials:

```bash
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
gcloud auth application-default login
```

See Google's [Application Default Credentials guide](https://cloud.google.com/docs/authentication/provide-credentials-adc). Override
`VEO_OPERATION_TIMEOUT_SECONDS`, `VEO_REQUEST_TIMEOUT_SECONDS`, or
`VEO_DOWNLOAD_TIMEOUT_SECONDS` when a deployment needs different limits.
See the
[video generation](../remote_video_generation/README.md),
[object flow extraction](../remote_object_flow/README.md), and
[hand flow](../remote_hand_flow/README.md) guides to install, forward, and verify each service.

## Verify

After installing and forwarding the three services, run the complete recorded
color-sorting example:

```bash
pixi run -e local-planning-full verify-closed-loop
```

This uses one Wan rollout per action to keep the installation check reasonably
small while exercising live planning, video generation, rollout selection,
object/hand-flow grounding, transform compilation, verification, and recovery
when triggered. Use the normal closed-loop command below for paper-scale
candidate counts or Veo.

For a shorter walkthrough, run one live-planned execution step
and use the recorded next-step camera state for transition verification:

```bash
pixi run -e local-planning-full color-sorting-step-live \
  --video_backend both \
  --recovery_video_backend both \
  --execution_num_video_per_action 2 \
  --recovery_num_videos 2 \
  --debug_flow_review
```

This command proposes the action and runs video prompting, generation, rollout
selection, geometric grounding, and state verification live. Only the simulated
post-execution camera observation is recorded: `step_002/start.png` stands in
for the state observed after executing step 1. The action may match or differ
from the action in the recorded trace. The VLM verifies the actual proposed
action against that observation and enters the normal recovery path only when
the transition fails.

Recovery action proposal, video generation, selection, and grounding are live.
Because the example data contains no captured result of that newly generated
recovery motion, the recorded-observation run pauses after recovery grounding
instead of reusing `step_002/start.png` as false post-recovery evidence. Online
execution resumes verification when the external controller supplies a fresh
post-recovery observation.

The task alias uses one Wan candidate when run without overrides to keep the
basic example inexpensive. The command above intentionally exercises both Wan
and Veo with two candidates per enabled backend. Both color-sorting aliases
invoke `run_closed_loop_execution.py` directly, so they
accept the same options shown by `closed-loop-execution --help`.

The `verify-video-server`, `verify-object-flow-server`, and `verify-hand-flow-server`
tasks provide isolated service diagnostics without reimplementing the
closed-loop pipeline.

For development, the fast workstation regression suite is grouped under one
command:

```bash
pixi run -e local-planning-full test-all
```

Additional diagnostic tasks are available in `pixi.toml`; they are optional and
are not part of the standard installation check.

## Run the Full Pipeline

`closed-loop-execution` is the primary entry point. It first determines task
coupling and horizon given the high-level task command. Coupled tasks use strategic beam search to choose a text
plan in advance; otherwise, the planner proposes actions reactively from each latest
observation at each step. Both modes regenerate execution-time videos, ground one selected
rollout, verify the returned observation, and recover before advancing if needed.

Each execution step follows the same contract:

1. Propose or read the current text action and `track_object`.
2. Generate Wan, Veo, or combined video rollouts from the latest observation.
3. Rank candidates using their final frames and 2D motion evidence.
4. Compute metric object flow, switching to calibrated hand flow only when the
   object transforms are unreliable or exceed `--flow_switch_theta_deg`.
5. Compile adjacent camera-frame relative transforms.
6. Request a fresh post-action observation and verify the intended transition.
7. Advance on success; otherwise generate, rank, ground, and verify recovery.

Inspect every available option with:

```bash
pixi run -e local-planning-full closed-loop-execution --help
```

### Recorded-observation showcase

The recorded example data is distributed separately from the source repository.
[Download and extract it using the example-data guide](../example_data/README.md).
The final path must be `example_data/color_sorting/`, without an extra nested
`color_sorting` directory. Generated outputs are written under `runs/`.

Run the recorded-observation showcase with:

```bash
pixi run -e local-planning-full closed-loop-execution \
  --goal "Put each block into the container of the matching color." \
  --input_frame example_data/color_sorting/start.png \
  --sample_root example_data/color_sorting \
  --execution_context recorded_observations \
  --video_backend wan22 \
  --recovery_video_backend wan22
```

This mode keeps task assessment, action proposal, video prompts, video generation,
rollout selection, object flow, hand flow, verification, and recovery VLM calls
live. It uses only the recorded next-step RGB-D start images in place of camera
observations after physical execution; it does not read `illustration_actions.json`.

The `--execution_context illustration` mode additionally replays
`illustration_actions.json` for deterministic example-data actions.

### Online execution for your own setup

After setting up all modules, run NovaPlan on your own RGB-D setup using:

```bash
pixi run -e local-planning-full closed-loop-execution \
  --goal "TASK DESCRIPTION" \
  --input_frame /path/to/my_sample/step_001/start.png \
  --sample_root /path/to/my_sample \
  --execution_context online \
  --observation_dir runs/my_run/external_observations \
  --observation_wait_seconds 3600
```

The initial `step_001/` directory must contain aligned `start.png`,
`start_depth.png`, and `config.json` files as described in the
[example-data guide](../example_data/README.md#prepare-an-initial-rgb-d-sample).
The accepted motion for each step is written by default to
`runs/closed_loop/<timestamp>/step_XXX/execution_step/relative_ee_transforms.npy`;
an explicit `--output_dir` replaces the `runs/closed_loop/<timestamp>` prefix.

After each normal or recovery trajectory, NovaPlan writes an
`observation_request.json`. The external system executes the referenced
relative transforms and writes `rgb.png`, aligned metric `depth.npy` (meters)
or integer `depth.png`, camera `config.json`, and finally an atomic `READY`
marker containing the same `request_id`. For `depth.png`, `depth_scale` is
meters per integer unit and defaults to `0.001`. RGB alone can support terminal
verification, but another grounding step requires aligned metric depth and
camera intrinsics.

## Planning and Execution Settings

The following arguments control beam search and execution settings:

```text
--beam_size
--num_action_per_beam
--num_video_per_action
--execution_num_action_per_step
--execution_num_video_per_action
--video_backend {wan22,veo3,both}
--recovery_video_backend {wan22,veo3,both}
--recovery_num_videos
```

Normal and grasp-recovery grounding are object-first. NovaPlan switches to hand flow when object flow is invalid or exceeds
`--flow_switch_theta_deg`. Rejected hand grounding triggers fresh video generation and
selection up to `--max_hand_flow_regenerations`.

Grasp recovery uses first-last-frame video generation toward the selected ideal
frame and follows the normal object-first grounding policy. Non-prehensile
recovery annotates one contact point and finger, then forces hand-centric
grounding. Both routes compile the accepted imagined motion into relative
end-effector transforms for closed-loop verification and recovery.

## Outputs and Review

Persistent outputs use one workflow-specific timestamped directory under
`runs/`. Direct closed-loop runs use `runs/closed_loop/<timestamp>/`,
planner-only runs use `runs/planner/<timestamp>/`, and installation checks use
`runs/verification/`. Explicit `--output_dir` or `--debug_dir` values override
these defaults. Unit tests use temporary directories and do not leave
persistent run artifacts.

Each closed-loop run tree contains concise
`terminal.log`, full `runtime_verbose.log`, selected rollouts, object/hand flow tracks, relative transforms, verification and recovery decisions, and
`closed_loop_summary.json`. Visual artifacts for debugging are grouped below
each `step_XXX/debug_artifacts/` directory:

```text
runs/closed_loop/<timestamp>/
  terminal.log
  runtime_verbose.log
  closed_loop_summary.json
  step_001/
    selected_rollout/
    execution_step/
      relative_ee_transforms.npy
      relative_ee_frame_indices.npy
      execution_step.npz
      execution_step.json
    debug_artifacts/
    recovery/attempt_001/
```

Entry zero of `relative_ee_transforms.npy` is identity. Later entries are
adjacent, left-multiplicative camera-frame deltas in meters:

```text
delta[t] = T_camera_motion[t] @ inverse(T_camera_motion[t - 1])
```

`relative_ee_frame_indices.npy` preserves the source video frames when tracking
skips frames. A calibrated downstream controller can map a delta into the robot
base frame with `T_base_camera @ delta_camera @ T_camera_base`.

For interactive visualization of the extracted object/hand flow after each normal or recovery grounding step:

```bash
pixi run -e local-planning-full closed-loop-execution ... --debug_flow_review
```

One persistent Viser server updates the current RGB-D point cloud and animates
cumulative object-flow trails or calibrated hand flows. The `Flow Layer`
control switches between object and hand artifacts when both exist. The GUI
waits for `Continue to next step` or `Stop execution`.

## Run Individual Stages

Strategic beam search only:

```bash
pixi run -e local-planning-full plan-beam-search \
  --goal "TASK DESCRIPTION" \
  --input_frame /path/to/start.png \
  --horizon 3
```

Compile an existing rollout and its flow into relative transforms:

```bash
pixi run -e local-planning-full compile-execution-step \
  --sample_dir example_data/color_sorting/step_001
```

If object flow is absent, add `--track_object`, `--object_flow_server_url`, and
`--hand_flow_server_url` so the compiler requests fresh grounding first. Outputs
are written under `test_results/execution_step/`.

During closed-loop execution, the selected transform sequence is followed by a
new observation, geometric re-grounding, state verification, and, when necessary, video-based recovery generation.

## Implementation Map

| Responsibility | Primary implementation |
| --- | --- |
| CLI and configuration | `run_closed_loop_execution.py` |
| Planning and rollout selection | `../novaplan/planner.py` |
| Closed-loop verification and recovery | `../novaplan/closed_loop_execution.py` |
| Observation handoff | `../novaplan/observations.py` |
| Relative-transform compilation | `run_execution_step.py`, `../novaplan/execution_step.py` |
