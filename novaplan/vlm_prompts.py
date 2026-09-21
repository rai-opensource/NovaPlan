# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Versioned VLM prompt contracts used by NovaPlan.

The paper's explicit Appendix prompts are the source of truth for action
proposal, rollout ranking, transition verification, and recovery.  The task
structure and video-prompt-extension contracts are implementation details not
specified verbatim in the paper; those retain the useful constraints from the
research ``main`` branch while using the paper's terminology and execution
semantics.

Keeping prompts here makes it possible to review the model-facing contract
without searching through API, parsing, and orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


PROMPT_CONTRACT_VERSION = "corl2026-v1"
CANONICAL_CONTACT_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class PromptContract:
    """Metadata for one model-facing prompt contract."""

    name: str
    version: str
    purpose: str
    provenance: tuple[str, ...]
    response_schema: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible prompt-contract metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "purpose": self.purpose,
            "provenance": list(self.provenance),
            "response_schema": dict(self.response_schema),
        }


TASK_STRUCTURE_SCHEMA = {
    "type": "object",
    "required": [
        "visual_analysis",
        "subtasks",
        "ordering_dependencies",
        "is_coupled",
        "plan_in_advance_allowed",
        "horizon",
        "coupling_reason",
        "reasoning",
    ],
    "properties": {
        "visual_analysis": {"type": "object"},
        "subtasks": {"type": "array"},
        "ordering_dependencies": {"type": "array", "items": {"type": "string"}},
        "is_coupled": {"type": "boolean"},
        "plan_in_advance_allowed": {"type": "boolean"},
        "horizon": {"type": "integer", "minimum": 0},
        "coupling_reason": {"type": "string"},
        "reasoning": {"type": "string"},
    },
}

ACTION_PROPOSAL_SCHEMA = {
    "type": "object",
    "required": ["phase", "dependency_analysis", "valid_objects", "proposals"],
    "properties": {
        "phase": {"type": "string"},
        "dependency_analysis": {"type": "string"},
        "valid_objects": {"type": "array", "items": {"type": "string"}},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "track_object", "reasoning"],
                "properties": {
                    "action": {"type": "string"},
                    "track_object": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
            },
        },
    },
}

ROLLOUT_RANKING_SCHEMA = {
    "type": "object",
    "required": ["rankings"],
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candidate_id", "success", "score", "reason"],
                "properties": {
                    "candidate_id": {"type": "integer", "minimum": 0},
                    "success": {"type": "boolean"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

TRANSITION_VERIFICATION_SCHEMA = {
    "type": "object",
    "required": ["success", "reason"],
    "additionalProperties": False,
    "properties": {
        "success": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

RECOVERY_SCHEMA = {
    "type": "object",
    "required": [
        "recovery_mode",
        "recovery_action",
        "mode_justification",
        "annotation",
        "prompt_P",
    ],
    "additionalProperties": False,
    "properties": {
        "recovery_mode": {"enum": ["grasp", "non_prehensile"]},
        "recovery_action": {"type": "string", "minLength": 1},
        "mode_justification": {
            "type": "object",
            "required": ["discrepancy_summary", "why_this_mode"],
        },
        "annotation": {
            "type": "object",
            "required": [
                "enabled",
                "object_name",
                "contact_point_definition",
                "edit_spec",
                "annotated_image_png_base64",
            ],
        },
        "prompt_P": {
            "type": "object",
            "required": ["enabled", "title", "text"],
        },
    },
}

ROLLOUT_SCORE_SCHEMA = {
    "type": "object",
    "required": ["confidence", "steps_to_goal"],
    "properties": {
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "steps_to_goal": {"type": "integer", "minimum": 0},
    },
}


PROMPT_REGISTRY: Mapping[str, PromptContract] = MappingProxyType(
    {
        "task_structure": PromptContract(
            name="task_structure",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Classify strategic versus reactive execution and count macro-actions.",
            provenance=(
                "Corl_2026_NovaPlan.pdf, Sec. 3 and Algorithm 1 (planning modes)",
                "NovaPlan implementation (no verbatim paper template)",
            ),
            response_schema=TASK_STRUCTURE_SCHEMA,
        ),
        "action_proposal": PromptContract(
            name="action_proposal",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Propose the next physically valid macro-action and tracked object.",
            provenance=(
                "Corl_2026_NovaPlan.pdf, Appendix task-decomposition/action-proposal prompt",
                "NovaPlan reference implementation commit 3c6ca75, action proposal prompt",
            ),
            response_schema=ACTION_PROPOSAL_SCHEMA,
        ),
        "video_prompt_extension": PromptContract(
            name="video_prompt_extension",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Turn a macro-action into a physically constrained Wan/Veo scene prompt.",
            provenance=(
                "NovaPlan reference implementation commit 3c6ca75, video prompt extension",
                "Corl_2026_NovaPlan.pdf, representative video prompts (Table 4)",
            ),
            response_schema={"type": "string", "minLength": 1},
        ),
        "rollout_ranking": PromptContract(
            name="rollout_ranking",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Rank generated rollout candidates using flow and final-state evidence.",
            provenance=(
                "Corl_2026_NovaPlan.pdf, Appendix rollout-ranking prompt",
                "NovaPlan reference implementation commit 3c6ca75, grid ranking prompt",
            ),
            response_schema=ROLLOUT_RANKING_SCHEMA,
        ),
        "transition_verification": PromptContract(
            name="transition_verification",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Verify a real transition against its generated target using three images.",
            provenance=(
                "Corl_2026_NovaPlan.pdf, Appendix three-image state-verification prompt",
            ),
            response_schema=TRANSITION_VERIFICATION_SCHEMA,
        ),
        "recovery_policy": PromptContract(
            name="recovery_policy",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Propose a grasp or one-contact non-prehensile FLF recovery action.",
            provenance=(
                "Corl_2026_NovaPlan.pdf, Appendix recovery prompt",
            ),
            response_schema=RECOVERY_SCHEMA,
        ),
        "rollout_score": PromptContract(
            name="rollout_score",
            version=PROMPT_CONTRACT_VERSION,
            purpose="Per-rollout heuristic retained for compatibility callers.",
            provenance=("NovaPlan reference implementation commit 3c6ca75, rollout scoring",),
            response_schema=ROLLOUT_SCORE_SCHEMA,
        ),
    }
)


def prompt_contract(name: str) -> PromptContract:
    """Return a named versioned prompt contract."""
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown VLM prompt contract {name!r}; expected one of {sorted(PROMPT_REGISTRY)}") from exc


def _history_text(action_history: Optional[Sequence[str]], previous_action: Optional[str]) -> str:
    history = [str(item).strip() for item in (action_history or ()) if str(item).strip()]
    if not history and previous_action:
        history = [str(previous_action).strip()]
    if not history:
        return "None (start of task)"
    return "\n".join(f"{index}. {action}" for index, action in enumerate(history, 1))


def build_task_structure_prompt(
    *,
    goal: str,
    min_horizon: int,
    max_horizon: int,
) -> str:
    """Build the strategic/reactive classifier and high-level horizon prompt."""

    return f"""You are NovaPlan's high-level task-structure assessor. Use only the supplied current RGB image.

GOAL: {goal}

1. Inventory every goal-relevant object and mark each visible subtask Done or Pending.
2. Identify direct ordering dependencies between pending macro-actions:
   - A task is COUPLED only when at least one pending macro-action must precede another because of a physical or semantic prerequisite visible in the scene.
   - Sharing one human hand, robot, workspace, or generic execution resource does NOT make otherwise independent subtasks coupled.
   - Do not record an edge whose successor must be grounded from state revealed only after its predecessor; that case is reactive, with ordering_dependencies=[].
   - If every pending subtask may be completed in any order, return ordering_dependencies=[] and is_coupled=false.
3. Determine observability before choosing a mode:
   - STRATEGIC / plan_in_advance_allowed=true only when the relevant state, destinations, and ordered dependencies are visible enough to propose a coherent multi-step text plan now.
   - REACTIVE / plan_in_advance_allowed=false when the task requires exploration or reveals new state after an action, or when independent subtasks should be re-grounded one at a time from the latest real observation.
   - Opening a closed drawer to discover or retrieve an initially hidden object is REACTIVE, because the post-open observation is required before choosing the next grounded action.
   - Visible assembly dependencies may be STRATEGIC when their order can be inferred from the current image.
4. Set is_coupled=true exactly when ordering_dependencies is non-empty. Coupled tasks use strategic planning; uncoupled tasks use reactive step-by-step planning.
5. Count distinct high-level video-action segments still required. One hand manipulates one separated object at a time. A visible one-object pick-and-place may be one segment. Do not count completed subtasks.
6. Return horizon=0 only if the goal is already complete; otherwise clamp it to [{min_horizon}, {max_horizon}].

Return JSON only:
{{
  "visual_analysis": {{
    "relevant_objects_detected": ["string"],
    "completed_subtasks": ["string"],
    "pending_subtasks": ["string"]
  }},
  "subtasks": [{{"index": 1, "description": "one macro-action", "target_object": "object"}}],
  "ordering_dependencies": [],
  "is_coupled": false,
  "plan_in_advance_allowed": false,
  "horizon": 1,
  "coupling_reason": "brief observability/dependency explanation",
  "reasoning": "brief count and mode explanation"
}}"""


ACTION_PROPOSAL_SYSTEM_PROMPT = (
    "You are an expert Single-Arm Robotic Planner controlling a Generative Video Model. "
    "Your commands must adhere to strict PHYSICS CAUSALITY and ROBOT MORPHOLOGY."
)


def build_action_proposal_prompt(
    *,
    goal: str,
    num_actions: int,
    previous_action: Optional[str] = None,
    action_history: Optional[Sequence[str]] = None,
    steps_remaining: Optional[int] = None,
    constraint_context: Optional[str] = None,
) -> str:
    """Build the paper/main action-proposal prompt with complete execution context."""

    if int(num_actions) < 1:
        raise ValueError("num_actions must be at least one")
    history = _history_text(action_history, previous_action)
    remaining = "Unknown" if steps_remaining is None else str(max(0, int(steps_remaining)))
    if steps_remaining is None:
        urgency = "The remaining horizon is unknown. Prioritize precision and safety."
    elif int(steps_remaining) <= 2:
        urgency = "Time is critical. Actions must be aggressive."
    else:
        urgency = "You have time. Prioritize precision and safety."
    constraints = (constraint_context or "None").strip() or "None"
    return f"""Analyze the current image and execution history. Propose EXACTLY {int(num_actions)} distinct narrative macro-actions.

GOAL: {goal}
FULL ACTION HISTORY:
{history}
REMAINING EXECUTION HORIZON: {remaining}
HORIZON GUIDANCE: {urgency}
CONSTRAINT CONTEXT: {constraints}

CRITICAL VISUAL ANALYSIS — INTERSECTIONS AND HIERARCHY:
1. Trace the target locations (slots, containers, stacking areas) for all relevant objects.
2. Detect whether target zones or required access paths cross or overlap.
3. Determine access and occlusion hierarchy:
   - If placing B would cover or bridge access to A's destination, A must be placed first.
   - Prefer deep, central, low, behind, or inside destinations before peripheral objects that narrow access.
4. Distinguish correctly placed objects from loose clutter. Do not move a completed object unless it blocks a necessary path.
5. Mentally simulate A-then-B and B-then-A. If only one order remains feasible, that prerequisite is mandatory. Every proposal must respect it; diversity may vary the safe approach, never violate the order.

PHASE AND PHYSICAL CONSTRAINTS:
1. Identify the current phase: Approaching, Interacting, Transporting, or Finishing.
2. No telekinesis: objects move only through visible hand contact. Use active causal verbs.
3. Use video-scale macro-actions, not numeric low-level commands. One visible-object pick-and-place may be one action.
4. The robot has one right hand. Never use bimanual actions or manipulate two separated objects in one proposal.
5. Do not complete a distant multi-stage task instantly. If the goal is visibly complete, use `FINISH: Hold position`.
6. Start each action directly with one definitive imperative verb. Do not mention "the robot", "the hand", alternatives with "or", or slash-separated choices.
7. For every proposal, name the primary `track_object` manipulated or contacted.

SELECTION FILTER:
- Propose only actions that pass the ordering and visibility tests.
- If multiple independent actions are currently valid, cover genuinely distinct valid branches.
- If a prerequisite exists, all {int(num_actions)} proposals must operate on that prerequisite.

Return JSON only, with EXACTLY {int(num_actions)} proposal objects:
{{
  "phase": "current phase and visible state",
  "dependency_analysis": "ordering simulation, rejected blockers, and critical prerequisite",
  "valid_objects": ["objects safe to manipulate now"],
  "proposals": [
    {{
      "action": "precise imperative macro-action",
      "track_object": "one primary object",
      "reasoning": "why this action is valid now"
    }}
  ]
}}"""


@dataclass(frozen=True)
class VideoPromptInstructions:
    """Build backend-specific video-generation prompt instructions."""
    backend: str
    language: str
    has_last_frame: bool
    system_prompt: str
    user_text: str
    positive_prefix: str
    negative_prompt: str


_BANNED_ACTOR_WORDS = (
    "robotic arm", "robot arm", "robot hand", "robot", "robotic", "gripper", "mechanical arm"
)


VIDEO_HARD_CONSTRAINT_PREFIX_ZH = (
    "【硬约束(必须逐条满足,不可省略)】"
    "1)只出现一只干净的普通人的右手(无手套/纹身/首饰),禁止第二只手/左手/机械手/机械臂/机器人手臂;"
    "2)除被手接触物体外其余物体与背景绝对静止,禁止幽灵移动;"
    "3)所有固体刚性不变:不变形/不弯曲/不变身,禁止凭空新增物体;"
    "4)手必须完全闭合抓握后再提起,接触关系连续稳定;"
    "5)对抓取动作,避免推/拖/扫滑行,优先抓取-提起-悬空移动-垂直放入;"
    "对明确的非抓取恢复动作,必须保留指定的唯一指尖、唯一接触点与连续接触规则;"
    "6)固定三脚架视角,无推拉摇移/无变焦/无剪辑,光照一致;"
    "7)动作连续无瞬移无跳帧,插入/放置需准确对齐并完全嵌入齐平;"
    "8)放置完成后,手部必须顺滑地完全消失在视野外,禁止停留在物体上方;"
    "9)手部动作严禁在抓取/放置关键帧遮挡物体中心或插槽开口,确保目标物体全程清晰可见;"
    "10)必须始终为单手操作:每一帧最多一只手、一个手腕和一条前臂,由同一只右手独立完成俯视单手抓取,禁止另一只手辅助。"
)


VIDEO_HARD_CONSTRAINT_PREFIX_EN = (
    "[HARD CONSTRAINTS (must satisfy each one, no exceptions)] "
    "1) Only one clean ordinary human right hand appears (no gloves/tattoos/jewelry), no second hand/left hand/mechanical hand/robotic arm; "
    "2) Except for objects touched by the hand, all other objects and background remain absolutely still, no ghost movement; "
    "3) All solid objects remain rigid: no deformation/bending/morphing, no objects appearing from nowhere; "
    "4) Hand must fully close grip before lifting, contact relationships continuous and stable; "
    "5) For grasp actions, avoid pushing/dragging/sweeping and prefer pick-lift-move through air-place vertically; "
    "for an explicitly non-prehensile recovery action, preserve its one named fingertip, one contact point, and continuous-contact rule; "
    "6) Fixed tripod view, no push-pull/pan/zoom/editing cuts, consistent lighting; "
    "7) Motion continuous with no teleportation or frame skipping, insertion/placement must be precisely aligned and fully seated flush; "
    "8) After placement is complete, the hand must smoothly exit and fully disappear from view, with no hovering above the object; "
    "9) The hand must not occlude the object center or slot opening during grasp/place keyframes, keeping the target visible throughout; "
    "10) The action is strictly one-handed in every frame: at most one hand, one wrist, and one forearm are visible, and the same right hand performs the top-down grasp without assistance."
)


def build_direct_video_prompt(*, backend: str, action: str) -> str:
    """Apply deterministic generation constraints without a VLM extension call."""

    _, language = _video_language(backend)
    action = str(action or "").strip()
    prefix = (
        VIDEO_HARD_CONSTRAINT_PREFIX_EN
        if language == "en"
        else VIDEO_HARD_CONSTRAINT_PREFIX_ZH
    )
    if action.startswith(prefix):
        return action
    label = "[ACTION]" if language == "en" else "【动作】"
    return f"{prefix}\n{label} {action}".strip()


def _video_language(backend: str) -> tuple[str, str]:
    value = (backend or "wan22").strip().lower()
    if value in {"wan", "wan22", "wan2.2", "comfy", "comfyui"}:
        return "wan22", "zh"
    if value in {"veo", "veo3", "veo3.1", "google"}:
        return "veo3", "en"
    raise ValueError(f"Unsupported video backend {backend!r}; expected Wan2.2 or Veo3.1")


def build_video_negative_prompt(backend: str) -> str:
    """Return the main-branch physical/visual negative prompt in backend language."""

    _, language = _video_language(backend)
    if language == "zh":
        return (
            "色调艳丽，变焦，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，多只手，三条腿，背景人很多，倒着走，"
            "机械臂，机器人手臂，机械手臂，机械手，机器人，机械爪，夹爪，"
            "第二只手，两只手，手套，纹身，首饰，幽灵移动，物体自动移动，背景物体移动，"
            "变形，弯曲，凭空出现的物体，瞬移，跳帧，不连续动作，推拉镜头，摇移，剪辑跳跃，"
            "手部遮挡物体中心，手部停留在物体上方"
        )
    return (
        "vibrant colors, zoom, overexposure, static, blurry details, subtitles, stylized, artwork, painting, "
        "still image, grayish overall, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, "
        "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, "
        "cluttered background, multiple hands, three legs, many people in background, walking backwards, "
        "robotic arm, robot arm, robot hand, robot, robotic, gripper, mechanical arm, second hand, two hands, "
        "gloves, tattoo, jewelry, ghost movement, objects moving by themselves, background objects moving, "
        "deformation, bending, objects appearing from nowhere, teleportation, frame skipping, discontinuous motion, "
        "push-pull shot, panning, editing jump, hand occluding object center, hand hovering above object"
    )


def build_video_prompt_extension_instructions(
    *,
    backend: str,
    action: str,
    goal: str = "",
    track_object: str = "",
    has_last_frame: bool = False,
) -> VideoPromptInstructions:
    """Build Wan-Chinese or Veo-English one/two-frame extension instructions."""

    canonical_backend, language = _video_language(backend)
    action = (action or "").strip()
    goal = (goal or "").strip()
    track_object = (track_object or "object").strip()
    negative = build_video_negative_prompt(canonical_backend)
    length_requirement = "150–250" if has_last_frame else "120–200"

    if language == "en":
        frame_semantics = (
            "You receive START and GOAL images. Infer one smooth physical trajectory that begins exactly at START and ends at GOAL. "
            "Treat GOAL as the required last-frame state, not a second scene to describe independently."
            if has_last_frame
            else
            "You receive one START image. Preserve its identity, geometry, camera, and untouched objects while describing one action."
        )
        system = f"""You are a video scene-description generator for Veo3.1. {frame_semantics}

HARD CONSTRAINTS:
1. Exactly one clean ordinary human right hand appears: natural skin texture, joint wrinkles, trimmed fingernails, and a shirt cuff; no left/second hand, gloves, tattoos, jewelry, robot, or gripper. Every frame contains at most one hand, one wrist, and one forearm. The same right hand performs a one-handed top-down grasp without assistance.
2. Every untouched object and the background remain absolutely stationary. Only an object in direct continuous hand contact may move.
3. Solid objects remain rigid and keep their identity; no deformation, bending, morphing, duplication, disappearance, or newly invented objects.
4. For grasping, fingers close fully before lifting; maintain stable contact while carrying; place with precise alignment and then release.
5. For an explicitly non-prehensile action, preserve exactly the stated fingertip/contact point and continuous-contact rule; do not rewrite it as a grasp.
6. Use one continuous fixed-tripod shot with stable lighting; no zoom, pan, tilt, shake, cuts, teleportation, or skipped motion.
7. Keep the target and destination visible. After completion, the hand retracts and fully exits; the final frame contains only stationary objects.

Output one English [SCENE DESCRIPTION] paragraph of {length_requirement} words. Do not explain these rules, emit JSON/XML, or use any of these actor words: {", ".join(_BANNED_ACTOR_WORDS)}."""
        user = (
            f"ACTION: {action}\nOVERALL GOAL: {goal or 'not provided'}\n"
            f"PRIMARY TRACKED OBJECT: {track_object}\nGenerate only the English scene description."
        )
    else:
        frame_semantics = (
            "你会收到首帧START与末帧GOAL。推演一段从START准确开始并在GOAL准确结束的连续物理轨迹；GOAL是必须满足的末帧状态。"
            if has_last_frame
            else
            "你会收到一张首帧START。保持其物体身份、几何、机位与未操作物体不变，只描述一个动作。"
        )
        system = f"""你是Wan2.2视频场景描述生成器。{frame_semantics}

【硬约束】
1. 画面中只能出现一只干净普通的人类右手：皮肤纹理、关节褶皱、整齐指甲与手腕衣袖清晰；禁止左手、第二只手、手套、纹身、首饰、机器人或夹爪。每一帧最多一只手、一个手腕和一条前臂，始终由同一只右手独立完成俯视单手抓取，禁止另一只手辅助。
2. 除被右手直接连续接触的目标外，其余物体与背景绝对静止；禁止幽灵移动。
3. 固体刚性与身份恒定：不变形、不弯曲、不变身、不复制、不消失、不凭空新增物体。
4. 抓取时手指先完全闭合再提起，搬运中接触稳定，放置时精确对齐后释放。
5. 若动作明确为非抓取接触，必须保留指定的唯一指尖、唯一接触点与连续接触规则，不得擅自改成抓取。
6. 固定三脚架单镜头，光照一致；无变焦、摇移、剪辑、瞬移或跳帧。
7. 目标与放置位置全程可见。完成后右手顺滑撤离并完全退出，末帧只留下静止物体。

只输出一段{length_requirement.replace('–', '至')}字中文【场景描述】，不要解释规则，不要JSON或XML，不得出现机械臂、机器人、夹爪等机械角色词。"""
        user = (
            f"动作：{action}\n总目标：{goal or '未提供'}\n"
            f"主要跟踪物体：{track_object}\n只生成中文场景描述。"
        )

    return VideoPromptInstructions(
        backend=canonical_backend,
        language=language,
        has_last_frame=bool(has_last_frame),
        system_prompt=system,
        user_text=user,
        positive_prefix=(
            VIDEO_HARD_CONSTRAINT_PREFIX_EN
            if language == "en"
            else VIDEO_HARD_CONSTRAINT_PREFIX_ZH
        ),
        negative_prompt=negative,
    )


def build_rollout_ranking_prompt(*, goal: str, action_descriptions: Sequence[str]) -> str:
    """Build the paper's grid-based rollout-ranking prompt."""

    descriptions = "\n".join(str(item) for item in action_descriptions)
    count = len(action_descriptions)
    return f"""Rank {count} robot rollouts. Goal: "{goal}"

IMAGE: Grid with {count} tiles. Each tile has TOP=motion flow overlay on initial frame, CYAN LINE=divider, BOTTOM=final frame. Yellow "ID: k" label on each tile.

Actions:
{descriptions}

SCORING RULES (Strict Hierarchical Check - apply in order, stop at first failure):
1. Target Check: Did the CORRECT object move? If the WRONG object moved or there is NO motion at all, set score=0.0.
2. Physics Check: Is physics realistic? If an object MELTS, DEFORMS, TELEPORTS, or a NEW object appears from nowhere, set score<=0.1.
3. Motion Check: Look at TOP. Does the motion direction/trajectory match the ACTION?
   - NO (wrong direction, drifted, stuck, or incomplete path): score<=0.2, success=false.
   - YES: continue.
4. Result Check: Look at BOTTOM. Does it show the expected outcome of the ACTION?
   - NO (deformed/new object, wrong final position, or incomplete action): score<=0.2, success=false.
   - YES (the final state matches the action and progresses toward the goal): score=0.3-1.0, success=true.

OUTPUT: JSON only. Score ALL candidates exactly once.
{{
  "rankings": [
    {{"candidate_id": 0, "success": true, "score": 0.9, "reason": "brief evidence-based reason"}}
  ]
}}
candidate_id is the zero-based ID shown in the yellow grid label."""


def build_transition_verification_prompt(*, action: str, goal: str) -> str:
    """Build the paper's three-image transition-verification prompt."""

    return f"""You are a Robotic Systems Verifier. Your task is to determine whether the CURRENT STATE (Image 2) satisfies the physical requirements of the intended ACTION, given the START STATE (Image 1).

INTENDED ACTION: {action}
GOAL SPECIFICATION: {goal}

You are provided with three images:
- Image 1 (START STATE): state before the action.
- Image 2 (CURRENT STATE): state after attempting the action.
- Image 3 (TARGET STATE): expected outcome from the video plan.

Verification logic:
1. Analyze the action and list its implied physical constraints; for example, insert requires being inside a cavity and flush, while place requires resting on a surface.
2. Compare Image 1 with Image 2. Identify what changed and whether the intended object moved into the expected position and state.
3. Inspect Image 2 and determine whether the action-specific physical constraints are satisfied.
4. Compare Image 2 with Image 3 as the visual reference and decide whether the object state is sufficiently close to the target outcome.

Return JSON only, with exactly these fields:
{{"success": boolean, "reason": "which physical constraints are or are not met, grounded in the three images"}}"""


def build_recovery_prompt(
    *,
    goal: str,
    current_state: str = "",
    target_state: str = "",
    previous_action: str = "",
    failure_reason: str = "",
    object_prompt: str = "object",
    action_history: Optional[Sequence[str]] = None,
) -> str:
    """Build the recovery-policy prompt used after a failed transition."""

    history = _history_text(action_history, previous_action)
    fingers = ", ".join(CANONICAL_CONTACT_FINGERS)
    return f"""You are a Recovery-Policy Controller for a manipulation task. The previous normal action failed verification. You must (i) select a recovery mode, (ii) propose one fresh, concise recovery action for the current observed state, and (iii) if non-prehensile recovery is chosen, spatially ground a single contact point, return an annotated image, and synthesize a structured prompt for video generation.

INPUTS:
- Current observation and image to annotate: Image 1
- Goal / target image: Image 2
- Goal text: {goal}
CURRENT STATE: {current_state or 'described by Image 1'}
TARGET STATE: {target_state or 'described by Image 2'}
FULL ACTION HISTORY:
{history}
FAILURE REASON: {failure_reason or 'not provided'}
OBJECT PROMPT: {object_prompt}

TASK 1 — RECOVERY MODE SELECTION:
Compare Image 1 with Image 2 and choose `grasp` or `non_prehensile`. Choose `non_prehensile` only when one small perturbation, such as a gentle poke or nudge, can plausibly correct the remaining discrepancy; otherwise choose `grasp`. Provide a brief justification grounded in visible evidence. Set `recovery_action` to the single executable correction you propose now. It must describe the selected mode, act on the current state in Image 1, and be concise enough to use for rollout ranking and transition verification. Do not merely copy the failed previous action.

TASK 2 — GRASP BEHAVIOR:
For `grasp`, make `recovery_action` a concrete grasp/regrasp correction. Do not annotate Image 1 and do not synthesize special prompt P. Set annotation.enabled=false, all other annotation fields null, prompt_P.enabled=false, and prompt_P title/text null. NovaPlan will generate first-last-frame recovery rollouts from Image 1 toward Image 2 using `recovery_action`.

TASK 3 — NON-PREHENSILE GROUNDING:
For `non_prehensile`, choose exactly one visible object, exactly one contact point on it, and exactly one canonical contact finger from: {fingers}. Put a small solid red star centered exactly on that point in Image 1 and return the annotated PNG as base64. End `annotation.contact_point_definition` with the pixel-center notation `PIXEL: [x, y]`. The text of prompt P must explicitly contain one line in the exact form `CONTACT FINGER: <finger>`, using one canonical value and mentioning no other finger.

For non-prehensile prompt P, elaborate the same correction stated in `recovery_action`; do not introduce a different action. Use labeled sections for GLOBAL CONSTRAINTS, SCENE DESCRIPTION, FRAMING CONSTRAINTS, CONTACT TARGET, HARD SPATIAL CONSTRAINTS, HAND APPEARANCE, HAND APPROACH, CONTINUOUS-CONTACT RULE, ACTION DESCRIPTION, and END BEHAVIOR. Require a single fixed-camera shot, exactly one contact point, exactly one push/poke, no new objects, no motion of untouched objects, and a final state matching Image 2.

Return strict JSON only, with exactly this top-level schema:
{{
  "recovery_mode": "grasp" | "non_prehensile",
  "recovery_action": "one fresh, concise action matching the selected recovery mode",
  "mode_justification": {{
    "discrepancy_summary": "string",
    "why_this_mode": "string"
  }},
  "annotation": {{
    "enabled": boolean,
    "object_name": "string" | null,
    "contact_point_definition": "string" | null,
    "edit_spec": {{
      "marker": "solid_red_star",
      "anchor": "center_of_star_is_contact_point",
      "placement": "string"
    }} | null,
    "annotated_image_png_base64": "string" | null
  }},
  "prompt_P": {{
    "enabled": boolean,
    "title": "string" | null,
    "text": "string" | null
  }}
}}"""


ROLLOUT_SCORE_SYSTEM_PROMPT = (
    "You are an expert single-hand manipulation evaluator. Analyze rollout frames and return only valid JSON."
)


def build_rollout_score_prompt(*, goal: str, action: str, frame_count: int) -> str:
    """Build the compatibility per-rollout scoring prompt."""

    return f"""Evaluate one manipulation rollout.
GOAL: {goal}
ACTION: {action}
Image 1 is the initial state; Images 2-{int(frame_count) + 1} are sequential rollout frames.
Assess progress, physical feasibility, final goal proximity, and remaining distinct macro-actions.
Return JSON only: {{"confidence": <float 0.0-1.0>, "steps_to_goal": <nonnegative integer>}}"""


def validate_action_proposal_payload(payload: Mapping[str, Any], *, expected_count: int) -> list[dict[str, str]]:
    """Validate and normalize the strict action-proposal response."""

    for key in ("phase", "dependency_analysis"):
        if not str(payload.get(key) or "").strip():
            raise ValueError(f"action proposal response has empty required field {key!r}")
    valid_objects = payload.get("valid_objects")
    if not isinstance(valid_objects, list) or any(not str(item).strip() for item in valid_objects):
        raise ValueError("action proposal response 'valid_objects' must be a list of non-empty strings")
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != int(expected_count):
        actual = len(proposals) if isinstance(proposals, list) else "non-list"
        raise ValueError(f"VLM returned {actual} proposals; expected exactly {int(expected_count)}")
    result: list[dict[str, str]] = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise ValueError(f"proposal {index} is not an object")
        normalized = {
            "action": str(proposal.get("action") or "").strip(),
            "track_object": str(proposal.get("track_object") or "").strip(),
            "reasoning": str(proposal.get("reasoning") or "").strip(),
        }
        missing = [key for key, value in normalized.items() if not value]
        if missing:
            raise ValueError(f"proposal {index} has empty required fields: {', '.join(missing)}")
        result.append(normalized)
    return result


def validate_ranking_payload(
    payload: Mapping[str, Any],
    *,
    candidate_count: Optional[int] = None,
    candidate_ids: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """Validate the paper ranking schema and one-result-per-candidate invariant."""

    if candidate_ids is None:
        if candidate_count is None:
            raise ValueError("candidate_count or candidate_ids is required")
        expected_ids = tuple(range(int(candidate_count)))
    else:
        expected_ids = tuple(int(value) for value in candidate_ids)
        if len(set(expected_ids)) != len(expected_ids) or any(value < 0 for value in expected_ids):
            raise ValueError("candidate_ids must be unique nonnegative integers")
        if candidate_count is not None and int(candidate_count) != len(expected_ids):
            raise ValueError("candidate_count does not match candidate_ids")
    expected_id_set = set(expected_ids)
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or len(rankings) != len(expected_ids):
        actual = len(rankings) if isinstance(rankings, list) else "non-list"
        raise ValueError(f"VLM returned {actual} rankings; expected exactly {len(expected_ids)}")
    seen: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for entry in rankings:
        if not isinstance(entry, Mapping):
            raise ValueError("ranking entry is not an object")
        candidate_id = int(entry.get("candidate_id", entry.get("candidate id", -1)))
        score = float(entry.get("score"))
        success = entry.get("success")
        reason = str(entry.get("reason") or "").strip()
        if candidate_id not in expected_id_set or candidate_id in seen:
            raise ValueError(f"invalid or duplicate candidate_id {candidate_id}")
        if not isinstance(success, bool):
            raise ValueError(f"candidate {candidate_id} success must be boolean")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"candidate {candidate_id} score must be in [0, 1]")
        if success and score < 0.3:
            raise ValueError(
                f"candidate {candidate_id} is marked successful but score {score} is below 0.3"
            )
        if not success and score > 0.2:
            raise ValueError(
                f"candidate {candidate_id} is marked unsuccessful but score {score} exceeds 0.2"
            )
        if not reason:
            raise ValueError(f"candidate {candidate_id} reason is empty")
        seen.add(candidate_id)
        normalized.append(
            {"candidate_id": candidate_id, "success": success, "score": score, "reason": reason}
        )
    return normalized
