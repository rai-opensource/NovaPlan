# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Provide OpenAI-compatible vision-language model adapters."""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import base64
import time
import re
from typing import List, Any, Optional

try:
    from .vlm_prompts import (
        ACTION_PROPOSAL_SYSTEM_PROMPT,
        ROLLOUT_SCORE_SYSTEM_PROMPT,
        build_action_proposal_prompt,
        build_rollout_ranking_prompt,
        build_rollout_score_prompt,
        build_task_structure_prompt,
        build_transition_verification_prompt,
        build_video_prompt_extension_instructions,
        prompt_contract,
        validate_action_proposal_payload,
        validate_ranking_payload,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from vlm_prompts import (
        ACTION_PROPOSAL_SYSTEM_PROMPT,
        ROLLOUT_SCORE_SYSTEM_PROMPT,
        build_action_proposal_prompt,
        build_rollout_ranking_prompt,
        build_rollout_score_prompt,
        build_task_structure_prompt,
        build_transition_verification_prompt,
        build_video_prompt_extension_instructions,
        prompt_contract,
        validate_action_proposal_payload,
        validate_ranking_payload,
    )

DEFAULT_OPENAI_VLM_MODEL = "gpt-5.6"
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class VLMAdapter:
    """Call an OpenAI-compatible vision-language model for NovaPlan decisions."""
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_VLM_MODEL,
        reasoning_effort: Optional[str] = None,
        task_models: Optional[dict] = None,
        task_reasoning_efforts: Optional[dict] = None,
    ):
        """
        Initialize the VLM adapter for OpenAI models.
        
        Args:
            model: Model to use:
                   - OpenAI: "gpt-5.6", "gpt-5.5", "gpt-4.1", etc.
                   If None, reads from NOVAPLAN_VLM_MODEL, then OPENAI_MODEL,
                   then DEFAULT_OPENAI_VLM_MODEL.
        
        Environment variables:
            OPENAI_API_KEY: Required for OpenAI models
            OPENAI_BASE_URL: Optional for OpenAI
            NOVAPLAN_VLM_MODEL: Optional planner VLM model override
            NOVAPLAN_VLM_REASONING_EFFORT: Optional default reasoning effort
        """
        self.model = model if model is not None else os.getenv("NOVAPLAN_VLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_VLM_MODEL))
        if self.model.startswith("gemini"):
            raise ValueError(
                "Gemini VLM models are disabled for NovaPlan planning. "
                f"Use an OpenAI model such as {DEFAULT_OPENAI_VLM_MODEL}."
            )
        self.reasoning_effort = self._normalize_reasoning_effort(
            reasoning_effort if reasoning_effort is not None else os.getenv("NOVAPLAN_VLM_REASONING_EFFORT")
        )
        self.task_models = self._build_task_models(task_models)
        self.task_reasoning_efforts = self._build_task_reasoning_efforts(task_reasoning_efforts)
        self.provider = "openai"
        self._init_openai()
        
        print(
            f"VLMAdapter initialized with model: {self.model}, provider: {self.provider}, "
            f"reasoning_effort={self.reasoning_effort or 'model-default'}"
        )
        if self.task_models:
            print(f"VLM task model overrides: {self.task_models}")
        if self.task_reasoning_efforts:
            print(f"VLM task reasoning efforts: {self.task_reasoning_efforts}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        from openai import OpenAI  # type: ignore
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI API mode")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    
    @staticmethod
    def _np_image_to_data_url(arr: np.ndarray) -> str:
        """Convert numpy array to data URL for OpenAI API."""
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    
    @staticmethod
    def _np_image_to_pil(arr: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        return Image.fromarray(arr)
    
    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # First, try to find JSON in markdown code blocks
        # Find the content between ```json and ``` or ``` and ```
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            # Try to find JSON object within the code block
            # Find balanced braces
            brace_count = 0
            start_idx = code_content.find('{')
            if start_idx != -1:
                for i in range(start_idx, len(code_content)):
                    if code_content[i] == '{':
                        brace_count += 1
                    elif code_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_candidate = code_content[start_idx:i+1]
                            try:
                                json.loads(json_candidate)
                                return json_candidate
                            except json.JSONDecodeError:
                                pass
        
        # Try to find JSON object directly in text (without code blocks)
        # Find balanced braces
        brace_count = 0
        start_idx = text.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found balanced JSON object
                        json_candidate = text[start_idx:i+1]
                        # Validate it's valid JSON by trying to parse it
                        try:
                            json.loads(json_candidate)
                            return json_candidate
                        except json.JSONDecodeError:
                            pass
        
        # If no JSON found, return original text (will fail parsing but that's expected)
        return text

    @staticmethod
    def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().lower()
        if not value or value in {"default", "model-default"}:
            return None
        if value not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort '{value}'. Expected one of {sorted(REASONING_EFFORTS)}.")
        return value

    def _build_task_models(self, task_models: Optional[dict]) -> dict:
        models = dict(task_models or {})
        env_map = {
            "horizon": ("NOVAPLAN_HORIZON_VLM_MODEL", "NOVAPLAN_HORIZON_MODEL", "OPENAI_HORIZON_MODEL"),
            "action": ("NOVAPLAN_ACTION_VLM_MODEL",),
            "prompt_extension": ("NOVAPLAN_PROMPT_EXTENSION_VLM_MODEL",),
            "ranking": ("NOVAPLAN_RANKING_VLM_MODEL",),
            "scoring": ("NOVAPLAN_SCORING_VLM_MODEL",),
            "verification": ("NOVAPLAN_VERIFICATION_VLM_MODEL",),
            "recovery": ("NOVAPLAN_RECOVERY_VLM_MODEL",),
        }
        for task, names in env_map.items():
            if models.get(task):
                continue
            for name in names:
                value = os.getenv(name)
                if value:
                    models[task] = value
                    break
        for task, model in list(models.items()):
            if model and str(model).startswith("gemini"):
                raise ValueError(f"Gemini model override for task '{task}' is disabled. Use an OpenAI model.")
        return {k: v for k, v in models.items() if v}

    def _build_task_reasoning_efforts(self, task_reasoning_efforts: Optional[dict]) -> dict:
        efforts = dict(task_reasoning_efforts or {})
        env_map = {
            "horizon": "NOVAPLAN_HORIZON_REASONING_EFFORT",
            "action": "NOVAPLAN_ACTION_REASONING_EFFORT",
            "prompt_extension": "NOVAPLAN_PROMPT_EXTENSION_REASONING_EFFORT",
            "ranking": "NOVAPLAN_RANKING_REASONING_EFFORT",
            "scoring": "NOVAPLAN_SCORING_REASONING_EFFORT",
            "verification": "NOVAPLAN_VERIFICATION_REASONING_EFFORT",
            "recovery": "NOVAPLAN_RECOVERY_REASONING_EFFORT",
        }
        for task, name in env_map.items():
            if efforts.get(task) is None and os.getenv(name):
                efforts[task] = os.getenv(name)
        # Horizon counting and prompt rewriting are bounded helper calls, so keep
        # them light unless the caller explicitly asks otherwise.
        efforts.setdefault("horizon", "low")
        efforts.setdefault("prompt_extension", "low")
        return {
            task: normalized
            for task, value in efforts.items()
            if (normalized := self._normalize_reasoning_effort(value)) is not None
        }

    def _model_for_task(self, task: Optional[str] = None) -> str:
        return self.task_models.get(task, self.model) if task else self.model

    def _reasoning_effort_for_task(self, task: Optional[str] = None) -> Optional[str]:
        if task and task in self.task_reasoning_efforts:
            return self.task_reasoning_efforts[task]
        return self.reasoning_effort

    def _openai_chat_kwargs(
        self,
        *,
        task: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        kwargs = {}
        model = self._model_for_task(task)
        # Some newer OpenAI models only accept the default temperature.
        if temperature is not None and not model.startswith("gpt-5"):
            kwargs["temperature"] = temperature
        if seed is not None and not model.startswith("gpt-5"):
            kwargs["seed"] = seed
        effort = self._normalize_reasoning_effort(reasoning_effort) if reasoning_effort is not None else self._reasoning_effort_for_task(task)
        if effort is not None and model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = effort
        return kwargs

    def _openai_chat_completion(
        self,
        *,
        task: str,
        messages: list,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        kwargs = self._openai_chat_kwargs(task=task, temperature=temperature, seed=seed)
        if response_format is not None:
            kwargs["response_format"] = response_format
        model = self._model_for_task(task)
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            if "reasoning_effort" not in kwargs:
                raise
            message = str(exc).lower()
            if "reasoning_effort" not in message and "unsupported" not in message:
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("reasoning_effort", None)
            print(
                f"OpenAI model {model} rejected reasoning_effort for task '{task}'; "
                "retrying with the model default."
            )
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                **retry_kwargs,
            )

    @staticmethod
    def _clean_generation_prompt(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        text = re.sub(r"^\s*(generation\s+prompt|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^\s*(?:\[\s*scene\s+description\s*\]|【\s*场景描述\s*】)\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return " ".join(text.split())

    @staticmethod
    def _sanitize_generation_prompt(text: str, *, language: str) -> str:
        """Remove mechanical actors that the prompt-extension model leaked."""

        replacement = (
            "有着清晰皮肤纹理的人类右手"
            if language == "zh"
            else "one clean ordinary human right hand"
        )
        banned = (
            "robotic arm",
            "mechanical arm",
            "robot arm",
            "robot hand",
            "mechanical hand",
            "robotic",
            "robot",
            "gripper",
            "机器人手臂",
            "机械手臂",
            "机械臂",
            "机械手",
            "机械爪",
            "夹爪",
            "机器人",
        )
        sanitized = str(text or "")
        for actor in sorted(banned, key=len, reverse=True):
            sanitized = re.sub(re.escape(actor), replacement, sanitized, flags=re.IGNORECASE)
        if language == "zh":
            sanitized = sanitized.replace("机械", "物理")
        return " ".join(sanitized.split()).strip()

    @staticmethod
    def _normalize_proposed_action(action: str) -> tuple[str, bool]:
        """Apply the paper/main single-hand action post-processing contract."""

        cleaned = re.sub(r"^\s*\d+\.\s*", "", str(action or "").strip())
        if re.match(r"^(?:FINISH|STOP)\b", cleaned, flags=re.IGNORECASE):
            return "Hold current position securely.", True

        cleaned = re.sub(r"^[\w\s\(\)\-]+:\s*", "", cleaned).strip()
        lowered = cleaned.lower()
        prefixes = (
            "the robot right hand ",
            "the robot left hand ",
            "the robot hand ",
            "robot right hand ",
            "robot left hand ",
            "robot hand ",
            "the robotic arm ",
            "the robot ",
            "robot ",
            "the human right hand ",
            "the human hand ",
            "the right hand ",
            "the hand ",
            "a hand ",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        if re.search(
            r"\b(?:both\s+hands|two\s+hands|second\s+hand|left\s+and\s+right\s+hands?)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            raise ValueError("action proposal violates the single-hand contract")
        cleaned = re.sub(r"\bhands\b", "hand", cleaned, flags=re.IGNORECASE)
        if re.search(
            r"\b(?:robot|robotic|gripper|mechanical\s+(?:arm|hand))\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            raise ValueError("action proposal still contains a forbidden mechanical actor")
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned, False

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "coupled"}:
                return True
            if lowered in {"false", "no", "0", "independent", "uncoupled", "not coupled"}:
                return False
        return default

    def estimate_horizon(
        self,
        image: np.ndarray,
        goal: str,
        fallback_horizon: int = 3,
        min_horizon: int = 1,
        max_horizon: int = 8,
    ) -> dict:
        """
        Classify task coupling, then estimate the high-level horizon.

        The prompt follows the old state-verification/action-planning style and
        only updates the API syntax to OpenAI chat.
        """
        fallback_horizon = max(1, int(fallback_horizon))
        min_horizon = max(1, int(min_horizon))
        max_horizon = max(min_horizon, int(max_horizon))
        frame_url = self._np_image_to_data_url(image)

        prompt = build_task_structure_prompt(
            goal=goal,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
        )

        try:
            t0 = time.time()
            resp = self._openai_chat_completion(
                task="horizon",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": frame_url, "detail": "high"}},
                    ],
                }],
                response_format={"type": "json_object"},
                temperature=0.2,
                seed=42,
            )
            horizon_model = self._model_for_task("horizon")
            horizon_effort = self._reasoning_effort_for_task("horizon") or "model-default"
            print(f"VLM horizon count (OpenAI {horizon_model}, effort={horizon_effort}) took {time.time()-t0:.2f}s")
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(self._extract_json_from_text(raw))
            required = {
                "visual_analysis",
                "subtasks",
                "ordering_dependencies",
                "is_coupled",
                "plan_in_advance_allowed",
                "horizon",
                "coupling_reason",
                "reasoning",
            }
            missing = required - set(data)
            if missing:
                raise ValueError(f"task-structure response is missing fields: {sorted(missing)}")
            if not isinstance(data["visual_analysis"], dict) or not isinstance(data["subtasks"], list):
                raise ValueError("task-structure visual_analysis/subtasks have invalid types")
            if not isinstance(data["ordering_dependencies"], list):
                raise ValueError("task-structure ordering_dependencies must be a list")
            if not isinstance(data["is_coupled"], bool) or not isinstance(data["plan_in_advance_allowed"], bool):
                raise ValueError("task-structure mode fields must be booleans")
            if isinstance(data["horizon"], bool) or not isinstance(data["horizon"], int):
                raise ValueError("task-structure horizon must be an integer")
            ordering_dependencies = []
            for dependency in data["ordering_dependencies"]:
                if not isinstance(dependency, str) or not dependency.strip():
                    raise ValueError("task-structure ordering dependencies must be non-empty strings")
                ordering_dependencies.append(dependency.strip())
            model_reported_is_coupled = data["is_coupled"]
            is_coupled = bool(ordering_dependencies)
            if model_reported_is_coupled != is_coupled:
                print(
                    "VLM task coupling correction: reported "
                    f"is_coupled={model_reported_is_coupled}, but "
                    f"ordering_dependencies={ordering_dependencies}; using is_coupled={is_coupled}."
                )
            data["ordering_dependencies"] = ordering_dependencies
            data["model_reported_is_coupled"] = model_reported_is_coupled
            plan_in_advance_allowed = is_coupled
            data["is_coupled"] = is_coupled
            data["plan_in_advance_allowed"] = plan_in_advance_allowed

            raw_horizon = data.get("horizon")
            horizon = int(raw_horizon if raw_horizon is not None else fallback_horizon)
            if horizon != 0:
                horizon = max(min_horizon, min(max_horizon, horizon))
            if is_coupled:
                planner_horizon = horizon
                reactive_execution_horizon = None
                execution_mode = "plan_in_advance"
                horizon_source = "vlm_count_coupled_plan_in_advance"
            else:
                planner_horizon = 0 if horizon == 0 else 1
                reactive_execution_horizon = horizon
                execution_mode = "reactive_step_by_step"
                horizon_source = "vlm_count_uncoupled_reactive_execution"
            data["horizon"] = horizon
            data["planner_horizon"] = planner_horizon
            data["reactive_execution_horizon"] = reactive_execution_horizon
            data["execution_mode"] = execution_mode
            data["auto_horizon"] = horizon
            data["horizon_source"] = horizon_source
            data["fallback_horizon"] = fallback_horizon
            data["min_horizon"] = min_horizon
            data["max_horizon"] = max_horizon
            data["model"] = horizon_model
            data["reasoning_effort"] = horizon_effort
            data["prompt_contract"] = prompt_contract("task_structure").metadata()
            return data
        except Exception as e:
            print(f"OpenAI horizon count error: {e}")
            return {
                "horizon": max(min_horizon, min(max_horizon, fallback_horizon)),
                "planner_horizon": max(min_horizon, min(max_horizon, fallback_horizon)),
                "reactive_execution_horizon": None,
                "auto_horizon": None,
                "horizon_source": "manual_fallback_error",
                "is_coupled": None,
                "plan_in_advance_allowed": True,
                "execution_mode": "plan_in_advance_fallback",
                "fallback_horizon": fallback_horizon,
                "min_horizon": min_horizon,
                "max_horizon": max_horizon,
                "model": self._model_for_task("horizon"),
                "reasoning_effort": self._reasoning_effort_for_task("horizon") or "model-default",
                "prompt_contract": prompt_contract("task_structure").metadata(),
                "error": str(e),
                "reasoning": "Fallback to configured horizon after horizon-counting failure.",
            }

    def extend_video_prompt(
        self,
        image: np.ndarray,
        action: str,
        goal: str = "",
        track_object: str = "",
        backend: str = "wan22",
        last_image: Optional[np.ndarray] = None,
    ) -> str:
        """
        Rewrite an action into a backend-specific generation prompt.

        Wan receives Chinese instructions and Veo receives English instructions.
        Supplying ``last_image`` switches the contract to first/last-frame
        semantics. The string return type is retained for existing callers.
        """
        action = (action or "").strip()
        if not action:
            raise ValueError("A non-empty action is required for video prompt extension")

        frame_url = self._np_image_to_data_url(image)
        instructions = build_video_prompt_extension_instructions(
            backend=backend,
            action=action,
            goal=goal,
            track_object=track_object,
            has_last_frame=last_image is not None,
        )

        try:
            t0 = time.time()
            user_content = [
                {"type": "image_url", "image_url": {"url": frame_url, "detail": "high"}},
            ]
            if last_image is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._np_image_to_data_url(last_image), "detail": "high"},
                    }
                )
            user_content.append({"type": "text", "text": instructions.user_text})
            resp = self._openai_chat_completion(
                task="prompt_extension",
                messages=[
                    {"role": "system", "content": instructions.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            prompt_model = self._model_for_task("prompt_extension")
            prompt_effort = self._reasoning_effort_for_task("prompt_extension") or "model-default"
            print(
                f"VLM prompt extension ({prompt_model}, backend={instructions.backend}, "
                f"language={instructions.language}, effort={prompt_effort}) took {time.time()-t0:.2f}s"
            )
            scene_description = self._clean_generation_prompt(resp.choices[0].message.content or "")
            scene_description = self._sanitize_generation_prompt(
                scene_description,
                language=instructions.language,
            )
            if not scene_description:
                raise ValueError("VLM returned an empty video-generation prompt")
            label = "[SCENE DESCRIPTION]" if instructions.language == "en" else "【场景描述】"
            return f"{instructions.positive_prefix}\n{label} {scene_description}"
        except Exception as e:
            raise RuntimeError(f"OpenAI video prompt extension failed: {e}") from e

    def propose_actions(
        self,
        image: np.ndarray,
        goal: str,
        num_actions: int,
        previous_action: str = None,
        steps_remaining: int = None,
        constraint_context: str = None,
        action_history: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Propose num_actions actions given current image and goal.

        Returns:
            List of dicts, each containing:
                - 'action': str - The action description for video generation
                - 'track_object': str - The object to track with SAM3/CoTracker
        """
        return self._propose_actions_openai(
            image,
            goal,
            num_actions,
            previous_action,
            steps_remaining,
            constraint_context,
            action_history,
        )

    def _propose_actions_openai(
        self,
        image: np.ndarray,
        goal: str,
        num_actions: int,
        previous_action: str = None,
        steps_remaining: int = None,
        constraint_context: str = None,
        action_history: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Propose actions using OpenAI.
        Returns list of dicts with 'action' and 'track_object' for each proposed action.
        """
        import json

        frame_url = self._np_image_to_data_url(image)
        prompt = build_action_proposal_prompt(
            goal=goal,
            num_actions=num_actions,
            previous_action=previous_action,
            action_history=action_history,
            steps_remaining=steps_remaining,
            constraint_context=constraint_context,
        )

        user = {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": frame_url, "detail": "high"}}
        ]}

        t0 = time.time()
        try:
            resp = self._openai_chat_completion(
                task="action",
                messages=[
                    {"role": "system", "content": ACTION_PROPOSAL_SYSTEM_PROMPT},
                    user,
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                seed=42,
            )

            action_model = self._model_for_task("action")
            action_effort = self._reasoning_effort_for_task("action") or "model-default"
            print(f"VLM propose_actions (OpenAI {action_model}, effort={action_effort}) took {time.time()-t0:.2f}s")
            
            content = resp.choices[0].message.content or "{}"
            data = json.loads(self._extract_json_from_text(content))
            proposals = validate_action_proposal_payload(data, expected_count=num_actions)

            print(f"[Planner Phase]: {data.get('phase', 'Unknown')}")
            print(f"[Planner Dependencies]: {data.get('dependency_analysis', 'None')}")
            print(f"[Planner Valid Objects]: {data.get('valid_objects', [])}")
            response_constraint_context = json.dumps(
                {
                    "phase": str(data.get("phase") or ""),
                    "dependency_analysis": str(data.get("dependency_analysis") or ""),
                    "valid_objects": [str(item) for item in data.get("valid_objects", [])],
                },
                sort_keys=True,
                separators=(",", ":"),
            )

            result = []
            for prop in proposals:
                action = prop["action"]
                track_object = prop["track_object"]
                cleaned, is_finish = self._normalize_proposed_action(action)
                if is_finish:
                    track_object = "human hand"
                
                result.append({
                    "action": cleaned,
                    "track_object": track_object,
                    "is_finish": is_finish,
                    "reasoning": prop["reasoning"],
                    "constraint_context": response_constraint_context,
                    "prompt_contract": prompt_contract("action_proposal").metadata(),
                })

            if any(not item["action"] for item in result):
                raise ValueError("VLM returned an empty action after normalization")
            print(f"[OpenAI] Proposed {len(result)} actions")
            return result

        except Exception as e:
            raise RuntimeError(f"OpenAI action proposal failed: {e}") from e

    def _parse_actions(self, text: str, num_actions: int) -> List[str]:
        """Strict legacy parser retained for API compatibility.

        Live proposal calls use the richer proposal schema. Compatibility
        callers receive an explicit error instead of a fabricated motion.
        """

        data = json.loads(self._extract_json_from_text(text))
        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list) or len(raw_actions) != int(num_actions):
            actual = len(raw_actions) if isinstance(raw_actions, list) else "non-list"
            raise ValueError(f"VLM returned {actual} actions; expected exactly {int(num_actions)}")
        actions = [str(action).strip() for action in raw_actions]
        if any(not action for action in actions):
            raise ValueError("VLM returned an empty action")
        return actions

    def score_rollout(self, image: np.ndarray, goal: str, action: str, rollout: np.ndarray) -> float:
        """Score a video rollout given the goal and action."""
        return self._score_rollout_openai(image, goal, action, rollout)

    def verify_transition(
        self,
        *,
        start_image: np.ndarray,
        current_image: np.ndarray,
        target_image: np.ndarray,
        action: str,
        goal: str,
    ) -> dict:
        """Verify the executed transition using the paper's three-image critic."""

        prompt = build_transition_verification_prompt(action=action, goal=goal)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": self._np_image_to_data_url(start_image), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": self._np_image_to_data_url(current_image), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": self._np_image_to_data_url(target_image), "detail": "high"}},
        ]
        t0 = time.time()
        try:
            resp = self._openai_chat_completion(
                task="verification",
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.2,
                seed=42,
            )
            model = self._model_for_task("verification")
            effort = self._reasoning_effort_for_task("verification") or "model-default"
            print(f"VLM transition verification ({model}, effort={effort}) took {time.time()-t0:.2f}s")
            data = json.loads(self._extract_json_from_text(resp.choices[0].message.content or "{}"))
            if set(data) != {"success", "reason"}:
                raise ValueError("verification response must contain exactly 'success' and 'reason'")
            if not isinstance(data.get("success"), bool):
                raise ValueError("verification response 'success' must be a boolean")
            if not str(data.get("reason") or "").strip():
                raise ValueError("verification response 'reason' must be a non-empty string")
            return {
                "success": data["success"],
                "reason": str(data.get("reason") or ""),
                "raw": data,
                "model": model,
                "reasoning_effort": effort,
                "prompt_contract": prompt_contract("transition_verification").metadata(),
            }
        except Exception as e:
            print(f"OpenAI transition verification error: {e}")
            return {
                "success": False,
                "reason": f"verification failed: {e}",
                "raw": {},
                "model": self._model_for_task("verification"),
                "reasoning_effort": self._reasoning_effort_for_task("verification") or "model-default",
                "prompt_contract": prompt_contract("transition_verification").metadata(),
            }

    def decide_recovery(
        self,
        context: Any,
        *,
        fallback_to_heuristic: bool = True,
    ) -> Any:
        """Propose a corrective action and choose its recovery mode from the supplied images."""

        try:
            from .recovery_policy import choose_recovery_strategy, recovery_decision_from_model_text
        except ImportError:
            from recovery_policy import choose_recovery_strategy, recovery_decision_from_model_text

        content = [{"type": "text", "text": context.to_prompt()}]
        for image_path in (context.current_image_path, context.target_image_path):
            if not image_path:
                continue
            try:
                image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            content.append({"type": "image_url", "image_url": {"url": self._np_image_to_data_url(image), "detail": "high"}})

        t0 = time.time()
        try:
            resp = self._openai_chat_completion(
                task="recovery",
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.2,
                seed=42,
            )
            model = self._model_for_task("recovery")
            effort = self._reasoning_effort_for_task("recovery") or "model-default"
            print(f"VLM recovery decision ({model}, effort={effort}) took {time.time()-t0:.2f}s")
            raw = resp.choices[0].message.content or "{}"
            decision = recovery_decision_from_model_text(raw, context)
            decision.metadata["model_raw"] = raw
            decision.metadata["model"] = model
            decision.metadata["reasoning_effort"] = effort
            decision.metadata["prompt_contract"] = prompt_contract("recovery_policy").metadata()
            return decision
        except Exception as exc:
            print(f"OpenAI recovery decision error: {exc}")
            if isinstance(exc, ValueError):
                raise RuntimeError(
                    "Recovery VLM response violated the grounding contract; reject it and regenerate "
                    "instead of guessing a contact finger or point."
                ) from exc
            if fallback_to_heuristic:
                decision = choose_recovery_strategy(context)
                decision.metadata["prompt_contract"] = prompt_contract("recovery_policy").metadata()
                decision.metadata["vlm_error"] = str(exc)
                return decision
            raise

    def _score_rollout_openai(self, image: np.ndarray, goal: str, action: str, rollout: np.ndarray) -> float:
        """Score rollout using OpenAI API."""
        T = rollout.shape[0]
        idxs = np.linspace(0, T-1, num=min(6, T), dtype=int)
        frame_urls = [self._np_image_to_data_url(rollout[i]) for i in idxs]
        
        system = {
            "role": "system", 
            "content": [{
                "type": "text", 
                "text": ROLLOUT_SCORE_SYSTEM_PROMPT,
            }]
        }
        
        user_text = build_rollout_score_prompt(goal=goal, action=action, frame_count=len(idxs))

        user_content: List[Any] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": self._np_image_to_data_url(image), "detail": "low"}}
        ]
        for u in frame_urls:
            user_content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})
        user = {"role": "user", "content": user_content}
        t0 = time.time()
        resp = self._openai_chat_completion(task="scoring", messages=[system, user])
        scoring_model = self._model_for_task("scoring")
        scoring_effort = self._reasoning_effort_for_task("scoring") or "model-default"
        print(f"VLM evaluate_actions ({scoring_model}, effort={scoring_effort}) took {time.time()-t0:.2f}s")
        text = resp.choices[0].message.content or ""
        return self._parse_score(text)
    
    def _parse_score(self, text: str) -> float:
        """Parse the registered rollout-score schema without optimistic defaults."""

        json_text = self._extract_json_from_text(text)
        data = json.loads(json_text)
        if not isinstance(data, dict):
            raise ValueError("rollout score response must be a JSON object")
        missing = {"confidence", "steps_to_goal"} - set(data)
        if missing:
            raise ValueError(f"rollout score response is missing fields: {sorted(missing)}")
        raw_steps = data["steps_to_goal"]
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps < 0:
            raise ValueError("steps_to_goal must be a nonnegative integer")
        raw_confidence = data["confidence"]
        if isinstance(raw_confidence, bool):
            raise ValueError("confidence must be a number in [0, 1]")
        confidence = float(raw_confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number in [0, 1]")
        return confidence - 0.1 * float(raw_steps)

    def rank_rollouts_batch(
        self, 
        goal: str,
        candidates: List[dict],  # Each: {'action': str, 'flow_image': np.ndarray or None}
        top_n: int = 4,
        debug_dir: Any = None,  # Optional Path to save grid images for debugging
        step: int = None  # Optional step number for filename
    ) -> List[dict]:
        """
        Rank multiple action candidates from generated video summaries in a single VLM call.
        
        Args:
            goal: The task goal
            candidates: Candidate dicts with ``action``, ``rollout``, and the
                CoTracker3 object ``flow_image`` required by Appendix C.2.
            top_n: Number of top candidates to return
            debug_dir: Optional Path to save grid images for debugging
            
        Returns:
            List of top_n candidates with 'score' added, sorted by score descending
        """
        return self._rank_rollouts_batch_openai(goal, candidates, top_n, debug_dir, step)

    def _create_image_grid(self, images: List[np.ndarray], labels: List[str] = None) -> Image.Image:
        """
        Helper to stitch multiple numpy images into a single labeled grid PIL Image.
        Draws high-contrast ID labels directly onto the pixels.
        """
        if not images:
            return None
        
        n = len(images)
        # Calculate optimal grid dimensions (approx square)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        h, w, c = images[0].shape
        
        # Create canvas
        grid_w = w * cols
        grid_h = h * rows
        grid_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid_img)
        
        # Try to use a larger font, otherwise default
        try:
            # Try loading a standard font, usually available on linux
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        except OSError:
            # Fallback for systems without that specific font
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except OSError:
                font = ImageFont.load_default()

        for i, img_arr in enumerate(images):
            r = i // cols
            c_idx = i % cols
            
            x_offset = c_idx * w
            y_offset = r * h
            
            # Paste image
            pil_img = Image.fromarray(img_arr)
            grid_img.paste(pil_img, (x_offset, y_offset))
            
            # Draw Label (Candidate ID and optionally score) clearly in top-left
            if labels and i < len(labels):
                label_text = labels[i]
                text_pos = (x_offset + 15, y_offset + 15)
                
                # Handle multi-line labels (e.g., "ID: 0\nScore: 0.95")
                lines = label_text.split('\n')
                line_height = 30  # Approximate line height
                
                # Calculate total bbox for all lines
                max_width = 0
                total_height = 0
                for line in lines:
                    bbox = draw.textbbox(text_pos, line, font=font)
                    max_width = max(max_width, bbox[2] - bbox[0])
                    total_height += line_height
                
                # Draw a black rectangle background for all text
                padded_box = (x_offset + 10, y_offset + 10, x_offset + 15 + max_width + 10, y_offset + 10 + total_height + 5)
                draw.rectangle(padded_box, fill="black")
                
                # Draw each line of text in bright yellow
                current_y = y_offset + 15
                for line in lines:
                    draw.text((x_offset + 15, current_y), line, font=font, fill=(255, 255, 0))
                    current_y += line_height
                
        return grid_img

    @staticmethod
    def _resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr).convert("RGB").resize(size, Image.Resampling.BILINEAR)

    def _create_candidate_review_tile(self, cand: dict) -> Optional[np.ndarray]:
        """
        Build a fixed-size visual summary for one candidate:
        top: flow image;
        cyan divider;
        bottom: final frame of the generated rollout.
        """
        rollout = cand.get("rollout")
        flow_image = cand.get("flow_image")
        if flow_image is None or not isinstance(rollout, np.ndarray) or len(rollout) == 0:
            return None

        tile_w, panel_h, divider_h = 640, 360, 8
        tile = Image.new("RGB", (tile_w, panel_h * 2 + divider_h), (255, 255, 255))

        top = self._resize_rgb(flow_image, (tile_w, panel_h))
        tile.paste(top, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, panel_h, tile_w, panel_h + divider_h], fill=(0, 255, 255))

        bottom = self._resize_rgb(rollout[-1], (tile_w, panel_h))
        tile.paste(bottom, (0, panel_h + divider_h))

        return np.array(tile, dtype=np.uint8)

    def _rank_rollouts_batch_openai(
        self,
        goal: str,
        candidates: List[dict],
        top_n: int,
        debug_dir: Any = None,
        step: int = None
    ) -> List[dict]:
        """Rank candidates using OpenAI API with a stitched grid for acceleration."""
        import json
        from pathlib import Path

        limit = max(0, int(top_n))
        if limit == 0:
            return []

        def _original_candidate_id(original_index: int, candidate: dict) -> int:
            value = candidate.get("candidate_id", original_index)
            if isinstance(value, bool):
                return original_index
            try:
                return int(value)
            except (TypeError, ValueError):
                return original_index
        
        # 1. Filter and Prepare Data
        valid_candidates = []
        review_images = []
        seen_candidate_ids: set[int] = set()
        for idx, cand in enumerate(candidates):
            review_image = self._create_candidate_review_tile(cand)
            if review_image is not None:
                candidate_id = _original_candidate_id(idx, cand)
                if candidate_id < 0 or candidate_id in seen_candidate_ids:
                    raise ValueError(f"candidate_id must be unique and nonnegative; got {candidate_id}")
                seen_candidate_ids.add(candidate_id)
                valid_candidates.append((idx, cand, candidate_id))
                review_images.append(review_image)
        
        if not valid_candidates:
            return []

        action_descriptions = []
        labels = []
        
        for orig_idx, cand, cid in valid_candidates:
            # Format: "ID <N>: <Action Text>"
            action_descriptions.append(f"ID {cid}: \"{cand.get('action', 'unknown')}\"")
            labels.append(f"ID: {cid}")

        # 2. Create Stitched Grid
        grid_image = self._create_image_grid(review_images, labels=labels)
        
        # Save grid image before ranking (for debugging)
        if debug_dir is not None:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            step_prefix = f"step_{step}_" if step is not None else ""
            grid_before_path = debug_path / f"{step_prefix}batch_ranking_grid_before.png"
            try:
                grid_image.save(grid_before_path)
                print(f"  💾 Saved grid image (before ranking) to {grid_before_path}")
            except Exception as e:
                print(f"  ⚠️ Failed to save grid image: {e}")
        
        grid_url = self._np_image_to_data_url(np.array(grid_image))

        # 3. Build Prompt. This follows Appendix C.2: flow overlay, cyan divider,
        # final frame, and a strict hierarchical scoring rubric.
        prompt_text = build_rollout_ranking_prompt(
            goal=goal,
            action_descriptions=action_descriptions,
        )
        
        # 4. API Call
        try:
            t0 = time.time()
            resp = self._openai_chat_completion(
                task="ranking",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": grid_url, "detail": "high"}}
                    ]}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            rank_model = self._model_for_task("ranking")
            rank_effort = self._reasoning_effort_for_task("ranking") or "model-default"
            print(f"VLM batch rank (OpenAI Grid {rank_model}, effort={rank_effort}) took {time.time()-t0:.2f}s")
            
            content = resp.choices[0].message.content
            data = json.loads(self._extract_json_from_text(content or "{}"))
            rankings = validate_ranking_payload(
                data,
                candidate_ids=[candidate_id for _, _, candidate_id in valid_candidates],
            )
            rankings = sorted(
                rankings,
                key=lambda entry: (-float(entry["score"]), int(entry["candidate_id"])),
            )
            
            # 5. Map results back
            all_results = []
            score_map = {}  # Map candidate_id -> score for annotation
            
            for rank_entry in rankings:
                cid = rank_entry.get("candidate_id", rank_entry.get("candidate id"))
                cid = int(cid)
                score = float(rank_entry.get("score", 0.1))
                score_map[cid] = score
                
                valid_entry = next(
                    (entry for entry in valid_candidates if entry[2] == cid),
                    None,
                )
                if valid_entry is not None:
                    orig_idx, orig_cand, original_candidate_id = valid_entry
                    cand_copy = dict(orig_cand)
                    cand_copy['candidate_id'] = original_candidate_id
                    cand_copy['grid_candidate_id'] = cid
                    cand_copy['success'] = bool(rank_entry["success"])
                    cand_copy['score'] = score
                    cand_copy['rank_reason'] = rank_entry["reason"]
                    cand_copy['prompt_contract'] = prompt_contract("rollout_ranking").metadata()
                    all_results.append(cand_copy)

            result = all_results[:limit]
            
            # Save grid image with scores annotated
            if debug_dir is not None:
                try:
                    # Create annotated grid with scores
                    annotated_labels = []
                    for orig_idx, cand, cid in valid_candidates:
                        score = score_map.get(cid, 0.0)
                        annotated_labels.append(f"ID: {cid}\nScore: {score:.3f}")
                    
                    annotated_grid = self._create_image_grid(review_images, labels=annotated_labels)
                    debug_path = Path(debug_dir)
                    step_prefix = f"step_{step}_" if step is not None else ""
                    grid_after_path = debug_path / f"{step_prefix}batch_ranking_grid_with_scores.png"
                    annotated_grid.save(grid_after_path)
                    print(f"  💾 Saved grid image (with scores) to {grid_after_path}")
                    
                    # Also save a JSON file with detailed ranking info
                    ranking_info = {
                        'goal': goal,
                        'step': step,
                        'num_candidates': len(valid_candidates),
                        'rankings': []
                    }
                    for rank_entry in rankings:
                        cid = int(rank_entry["candidate_id"])
                        valid_entry = next(
                            (entry for entry in valid_candidates if entry[2] == cid),
                            None,
                        )
                        if valid_entry is not None:
                            orig_idx, orig_cand, original_candidate_id = valid_entry
                            ranking_info['rankings'].append({
                                'candidate_id': original_candidate_id,
                                'grid_candidate_id': cid,
                                'success': bool(rank_entry["success"]),
                                'original_index': orig_idx,
                                'action': orig_cand.get('action', 'unknown'),
                                'generation_prompt': orig_cand.get('generation_prompt'),
                                'backend': orig_cand.get('backend', 'unknown'),
                                'score': float(rank_entry["score"]),
                                'reason': rank_entry["reason"]
                            })
                    
                    ranking_json_path = debug_path / f"{step_prefix}batch_ranking_results.json"
                    with open(ranking_json_path, 'w') as f:
                        json.dump(ranking_info, f, indent=2)
                    print(f"  💾 Saved ranking results to {ranking_json_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to save annotated grid/image: {e}")
            
            return result

        except Exception as e:
            raise RuntimeError(f"OpenAI rollout ranking failed: {e}") from e
