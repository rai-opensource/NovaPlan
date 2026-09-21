# [CoRL 2026] NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning

**Authors:** [Jiahui Fu*](https://jiahui-fu.github.io/), [Junyu Nan*](https://www.ri.cmu.edu/ri-people/junyu-nan/), [Lingfeng Sun](https://lingfeng.moe/), [Hongyu Li](https://lhy.xyz/), [Jianing Qian](https://scholar.google.com/citations?user=o67NTxYAAAAJ&hl=en), [Yilun Du](https://yilundu.github.io/), [Jennifer L. Barry](https://www.linkedin.com/in/jennifer-barry-742a0799/), [Kris Kitani](https://www.ri.cmu.edu/ri-faculty/kris-m-kitani/), [George Konidaris](https://cs.brown.edu/people/gdk/)

**Affiliations:** Robotics and AI Institute, Carnegie Mellon University, Brown University, University of Pennsylvania, Harvard University  

**Equal contribution**






**NovaPlan** is a hierarchical framework for zero-shot long-horizon manipulation. A VLM planner decomposes tasks and filters dynamically inconsistent futures by verifying multiple candidate video rollouts. NovaPlan translates selected futures into relative end-effector transforms by adaptively switching between object-centric and human hand flow, then closes the loop with outcome verification and local video-based recovery.

## ✨ Key Features

- **Closed-Loop Video Language Planning**: Propose, verify, and repair long-horizon visual plans from updated observations.
- **Multiple Visual Futures**: Generate Wan 2.2 and Veo candidates and select physically and semantically consistent transitions with a VLM.
- **Hybrid Geometric Grounding**: Switch between object-centric TAPIP3D flow and calibrated HaMeR hand flow when occlusion or depth artifacts make one reference unreliable.
- **Local Recovery**: Generate and rank recovery videos, including non-prehensile fingertip corrections.
- **Geometric Action Generation**: Convert selected object- or hand-centric flow into relative end-effector transforms.



## 📋 Abstract

Solving complex long-horizon robotic tasks requires joint reasoning over abstract task structure and low-level physical interaction. While combining Vision-Language Models (VLMs) and video generation models offers a promising path for zero-shot planning, their individual tendencies to hallucinate physics or violate geometric consistency often compound over time, preventing reliable real-world execution. We introduce **NovaPlan**, a hierarchical framework that enables robust, zero-shot long-horizon manipulation by systematically proposing, verifying, and repairing visual plans. At the high level, a VLM planner decomposes tasks and filters out dynamically inconsistent futures by verifying multiple candidate video rollouts. To translate these imagined futures into reliable physical actions, NovaPlan utilizes a hybrid geometric representation that adaptively switches between object-centric flow and human hand flow. Finally, NovaPlan closes the loop by continuously monitoring execution to verify outcomes and synthesize local, non-prehensile corrective behaviors, such as fingertip poking, when failures occur. Across diverse multi-stage tasks, NovaPlan substantially outperforms prior zero-shot systems, achieving complex assembly and dexterous error recovery entirely without task-specific training or demonstrations.

## 🚀 Getting Started



### Prerequisites

- **Workstation**: Python 3.10 or 3.11 and [Pixi](https://pixi.sh/).
- **Remote GPU hosts**: Linux, Git, Pixi, outbound HTTPS access, sufficient disk space for model weights, standard process/network tools (`procps`, `util-linux`, and `iproute2`), and an NVIDIA driver compatible with the selected PyTorch/CUDA build.
- **Video host**: A CUDA GPU for Wan 2.2, or Google Cloud credentials for Veo.
- **Object-flow host**: A CUDA GPU with MoGe2, CVD/RAFT, SAM3, and vanilla TAPIP3D assets.
- **Hand-flow host**: [HaMeR](https://github.com/geopavlakos/hamer) and a separately licensed MANO hand model.

As a reference deployment, we run the video service on eight H100 GPUs with
one ComfyUI worker per GPU, and the object-flow service on one H100 GPU.
These counts are an example for parallel throughput, not minimum hardware
requirements; smaller deployments are supported. See the
[video-service](remote_video_generation/README.md) and
[object-flow service](remote_object_flow/README.md) setup guides for worker
configuration.

### Installation

1. Clone the repository and install the full workstation environment:
  ```bash
   git clone https://github.com/rai-opensource/NovaPlan.git
   cd NovaPlan
   pixi install -e local-planning-full
  ```
   When a service runs on a separate GPU host, clone the same repository on
   that host. Do not install the workstation environment there; the first
   `pixi run -e <service-environment> ...` command in its setup guide creates
   the appropriate host environment.
2. Set up the modules needed for your run:

  | Module                     | Setup and verification                                           |
  | -------------------------- | ---------------------------------------------------------------- |
  | Closed-loop workstation    | [Local planning guide](local_planning/README.md)                 |
  | Wan video service          | [Video-generation guide](remote_video_generation/README.md)      |
  | Metric object flow service | [Object-flow guide](remote_object_flow/README.md)                |
  | Hand-flow grounding        | [Hand-flow service guide](remote_hand_flow/README.md)            |
  | Recorded example data      | [Example-data download and format guide](example_data/README.md) |

3. Follow the [example-data guide](example_data/README.md) to download the
  recorded color-sorting data and extract it to
   `example_data/color_sorting/`.
4. Inspect the closed-loop command-line interface:
  ```bash
   pixi run -e local-planning-full closed-loop-execution --help
  ```
   To prepare a new RGB-D scene and obtain camera-frame relative
   end-effector transforms, follow
   [Online execution for your own setup](local_planning/README.md#online-execution-for-your-own-setup).
5. Run the recorded color-sorting illustration. Example actions and
  observations are replayed from `example_data/color_sorting`, while video
   rollout generation and selection, geometric grounding, verification, and
   recovery run normally:
   Candidate counts apply per enabled backend, so this example generates eight
   Wan and eight Veo candidates for each normal or recovery rollout batch.
   Change `--execution_context` to `recorded_observations` to propose actions
   live while replaying recorded post-execution observations. Use `online` for
   a fully live run that waits for newly captured observations after execution.

Persistent generated artifacts are kept under `runs/`: direct closed-loop runs
use `runs/closed_loop/<timestamp>/`, planner-only runs use
`runs/planner/<timestamp>/`, and module installation checks use
`runs/verification/`. Unit tests use
temporary directories and do not leave persistent outputs.

## 📖 Pipeline Overview

NovaPlan turns an initial observation and task command into verified visual rollouts and relative end-effector transforms:

1. **Task Planning**: A VLM estimates the horizon and proposes a high-level action sequence.
2. **Video Rollouts**: Wan 2.2 and Veo generate multiple candidate futures from the latest observation.
3. **VLM Selection**: Candidate videos are ranked for semantic progress and physical consistency.
4. **Metric Depth**: The single public depth path runs MoGe2, optional CVD, and robust affine calibration against the first-frame sensor depth.
5. **Flow Grounding**: SAM3 and vanilla TAPIP3D recover object-centric 3D flow; HaMeR provides hand-centric flow when required.
6. **Transform Compilation**: The accepted flow is converted to relative end-effector transforms.
7. **Verification and Recovery**: A new observation is verified against the intended transition; failed steps trigger video-based recovery and re-grounding.

Paper settings use beam width 2, two action proposals per beam, and four video
rollouts per action during strategic planning. Actual execution regenerates
eight candidates per enabled backend from the latest observation. The
closed-loop CLI defaults to eight recovery candidates per enabled backend;
values above one enable flow-based recovery ranking.

### Repository Layout


| Path                                                 | Purpose                                                                                                    |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [local_planning/](local_planning/)                   | Workstation planning, closed-loop orchestration, relative-transform compilation, tests, and visualization. |
| [novaplan/](novaplan/)                               | Shared planning, video/flow clients, verification, recovery, and transform-compilation package.            |
| [remote_video_generation/](remote_video_generation/) | Wan 2.2/ComfyUI video-generation service and selection-flow support.                                       |
| [remote_object_flow/](remote_object_flow/)           | MoGe2 depth calibration, SAM3, vanilla TAPIP3D, and object-flow service.                                   |
| [remote_hand_flow/](remote_hand_flow/)               | Hand-flow service setup and API adapter, using HaMeR as its reconstruction backend.                        |




## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

The MIT License applies to original NovaPlan code and documentation.
Third-party components retain their respective licenses and terms.

## 🙏 Acknowledgments

NovaPlan builds upon several research projects, models, and open-source implementations:

- **[NovaFlow](https://github.com/rai-opensource/NovaFlow)**: Actionable-flow grounding for zero-shot manipulation.
- **[Wan 2.2](https://github.com/Wan-Video/Wan2.2)** and **[Veo](https://deepmind.google/models/veo/)**: Video generation.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** and **[LightX2V](https://github.com/ModelTC/lightx2v)**: Video-generation inference and workflow infrastructure.
- **[MoGe2](https://github.com/microsoft/MoGe)** and **[Mega-SAM](https://github.com/mega-sam/mega-sam)**: Metric depth and the CVD temporal-refinement components.
- **[SAM3](https://github.com/facebookresearch/sam3)** and **[TAPIP3D](https://github.com/zbw001/TAPIP3D)**: Object segmentation and 3D point tracking.
- **[HaMeR](https://github.com/geopavlakos/hamer)** and **[MANO](https://mano.is.tue.mpg.de/)**: Hand reconstruction and hand models.
- **[CoTracker3](https://github.com/facebookresearch/co-tracker)**: 2D point tracking used by selection/debugging components.



## 📚 Citations

If you find NovaPlan useful in your research, please cite our paper:

```bibtex
@article{fu2026novaplan,
  title={NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning},
  author={Fu, Jiahui and Nan, Junyu and Sun, Lingfeng and Li, Hongyu and Qian, Jianing and Du, Yilun and Barry, Jennifer L. and Kitani, Kris and Konidaris, George},
  journal={arXiv preprint arXiv:2602.20119},
  year={2026}
}
```

