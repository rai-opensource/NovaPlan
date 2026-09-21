# Example Data

The recorded `color_sorting` example demonstrates NovaPlan's module checks and
closed-loop pipeline using one three-step RGB-D task trace. The large images,
videos, and precomputed arrays are distributed separately; this source
directory intentionally contains only their setup and data-contract guide.

NovaPlan-authored example files are released under the repository's MIT
License. This does not cover third-party media. Only project-owned recordings
and generated assets cleared for public release should be included in the
download.

## Download the Recorded Example

[Download the color-sorting example from Google Drive](https://drive.google.com/drive/folders/1tbpqpVDBtpAWLwDT_n-zUSs2ZP2IN6Uj?usp=sharing),
then extract or copy the `color_sorting` directory to:

```text
<novaplan-repository>/example_data/color_sorting/
```

The resulting directory must contain `start.png` directly under
`color_sorting`; avoid an extra nested `color_sorting/color_sorting` level.

## Downloaded Layout

```text
example_data/color_sorting/
  goal.txt
  start.png
  start_depth.png
  intrinsics.json
  mask_prompts.json
  illustration_actions.json
  step_001/
    start.png
    start_depth.png
    rgb_video_16fps.mp4
    end_frame_rgb.png
    end_frame_depth.png
    config.json
    test_results/
      tapip3d_output.npz
  step_002/
    start.png
    start_depth.png
    rgb_video_16fps.mp4
    end_frame_rgb.png
    end_frame_depth.png
    config.json
  step_003/
    ...same files as step_002...
```

The downloaded files have these roles:

- `goal.txt`, `mask_prompts.json`, and `illustration_actions.json` describe the
  recorded task and its labeled transitions.
- Each `step_XXX/` contains an aligned starting RGB-D observation, its camera
  configuration, a recorded rollout, and the rollout's target frame.
- `step_001/test_results/tapip3d_output.npz` is required only for the offline
  transform-compilation example. Live object-flow extraction computes a fresh
  result instead.
- `end_frame_rgb.png` is a generated rollout target, not a captured
  post-execution observation.

## Execution Contexts

`illustration_actions.json` contains one `action` and `track_object` pair per
recorded transition. Use `--execution_context illustration` to replay those
actions and the recorded observations while running video generation,
selection, grounding, verification, and recovery normally.

Use `--execution_context recorded_observations` to keep VLM action proposal
live and replay only the post-execution RGB-D observations. Use `online` with a
fresh observation source instead of this recorded example.

The canonical commands for these modes are documented in the
[local-planning guide](../local_planning/README.md).

## Prepare an Initial RGB-D Sample

To run NovaPlan on a different scene, prepare the initial sample as:

```text
my_sample/
  step_001/
    start.png
    start_depth.png
    config.json
```

Use `my_sample/step_001/start.png` as `--input_frame` and `my_sample` as
`--sample_root`. The three files have the following contract:

- `start.png` is an RGB image.
- `start_depth.png` is a single-channel integer depth image registered and
  pixel-aligned to `start.png`.
- `config.json` contains the intrinsics of the aligned depth image and the
  number of meters represented by one depth-image unit.

A minimal camera configuration is:

```json
{
  "depth_scale": 0.001,
  "intrinsics": {
    "fx": 645.8,
    "fy": 645.8,
    "cx": 644.6,
    "cy": 364.9
  }
}
```

`fx`, `fy`, `cx`, and `cy` are pixel units. `depth_scale` is meters per integer
depth unit and defaults to `0.001` when omitted. For observations returned
during online execution, `depth.npy` may be supplied instead; its values must
already be metric meters.

See
[Online execution for your own setup](../local_planning/README.md#online-execution-for-your-own-setup)
for the command, subsequent-observation handoff, and generated transform path.

## Verify the Download

After installing the workstation environment, verify the bundled offline flow
artifact without contacting a remote service:

```bash
pixi run -e local-planning-full compile-execution-step \
  --sample_dir example_data/color_sorting/step_001 \
  --out_dir runs/verification/execution_step
```

After starting or forwarding the video, flow, and hand-flow services, run the
complete installation check:

```bash
pixi run -e local-planning-full verify-closed-loop
```

## Data Contract

- RGB and depth images must have the same width and height and be pixel-aligned.
- Camera configuration must provide `fx`, `fy`, `cx`/`cy` or `ppx`/`ppy`, and
  `depth_scale` when applicable.
- `start_depth.png` is aligned with the corresponding `start.png`.
- `end_frame_rgb.png` is a generated rollout target, not evidence of physical
  execution.
- Each recorded next-step `start.png` is a stand-in camera observation for the
  preceding transition.

Treat `example_data/color_sorting/` as read-only. Generated execution,
verification, visualization, and recovery artifacts belong under `runs/`, not
inside `example_data/`.
