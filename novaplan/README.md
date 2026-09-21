# NovaPlan Python Package

This package holds shared code used by the workstation scripts and remote
clients.

## Install

Install it editable with the workstation environment from the repository root:

```bash
pixi install -e local-planning
```

## Main Modules

- `closed_loop_execution.py`: in-process implementation of closed-loop
  video-language planning. It first consumes the planner's task-structure
  assessment, then runs strategic or reactive execution.
- `planner.py`: task-structure assessment, strategic beam search, one-step
  reactive planning, and VLM ranking.
- `video_generation/`: the unified video-generation API, with explicit Wan,
  Veo, and parallel hybrid clients plus shared result types.
- `vlm_prompts.py` and `llm_client.py`: versioned prompt contracts
  and the OpenAI/ChatGPT adapter for task structure, action proposal,
  backend-specific Chinese Wan and English Veo prompt extension, ranking,
  verification, and recovery.
- `flow_extraction_client.py`: client for lightweight 2D flow extraction from
  generated videos.
- `flow_switching.py`: object-flow quality checks and hand/object flow
  selection.
- `execution_step.py`: converts object/hand flow into camera-frame relative
  end-effector transforms for one execution step.
- `recovery_policy.py`: paper-style recovery-mode and annotation policy.
- `observations.py`: explicit recorded-example-data, external-filesystem, and
  callback handoff for post-execution RGB-D observations.
- `hand_flow/`: hand-flow service client, metric calibration, and hand-reference selection.

## Module Use

- Remote service launch/setup instructions live in `remote_*` folders.
- Workstation commands live in [local_planning/README.md](../local_planning/README.md).
- Package modules are import-safe and connect to remote services only when their
  workflows are invoked.

## Test

Run the package's fast workstation tests with the aggregate test command
documented by the local-planning module:

```bash
pixi run -e local-planning-full test-all
```
