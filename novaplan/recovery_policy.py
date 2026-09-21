#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""NovaPlan recovery policy helpers.

After a failed transition, the recovery policy proposes a fresh action and
chooses whether to realize it with grasp or non-prehensile grounding. For
non-prehensile corrections, a contact point and detailed generation prompt are
also grounded and the low-level execution is guided by hand flow. This module
exposes that decision as structured data without including robot arm control.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image, ImageDraw
import numpy as np

try:
    from .vlm_prompts import (
        CANONICAL_CONTACT_FINGERS,
        build_recovery_prompt,
        prompt_contract,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from vlm_prompts import (
        CANONICAL_CONTACT_FINGERS,
        build_recovery_prompt,
        prompt_contract,
    )


GRASP = "grasp"
NON_PREHENSILE = "non_prehensile"
VALID_MODES = {GRASP, NON_PREHENSILE}


@dataclass
class RecoveryContext:
    """Describe the failed execution state used for recovery selection."""
    goal: str
    current_state: str = ""
    target_state: str = ""
    previous_action: str = ""
    failure_reason: str = ""
    object_prompt: str = "object"
    current_image_path: Optional[str] = None
    target_image_path: Optional[str] = None
    planner_summary: Optional[Dict[str, Any]] = None

    def to_prompt(self) -> str:
        """Format the recovery context for a VLM prompt."""
        history = None
        if self.planner_summary:
            history = self.planner_summary.get("history")
            if not history:
                best = self.planner_summary.get("best_beam", {})
                history = best.get("actions", []) if isinstance(best, dict) else None
        return build_recovery_prompt(
            goal=self.goal,
            current_state=self.current_state,
            target_state=self.target_state,
            previous_action=self.previous_action,
            failure_reason=self.failure_reason,
            object_prompt=self.object_prompt,
            action_history=history,
        )


@dataclass
class RecoveryDecision:
    """Describe the selected recovery mode and contact annotation."""
    mode: str
    object_prompt: str
    reason: str
    confidence: float = 0.0
    contact_prompt: str = ""
    recovery_prompt: str = ""
    contact_point_2d: Optional[list[float]] = None
    annotation_enabled: bool = False
    annotation_object_name: Optional[str] = None
    contact_point_definition: Optional[str] = None
    annotation_edit_spec: Dict[str, Any] = field(default_factory=dict)
    annotated_image_png_base64: Optional[str] = None
    prompt_title: Optional[str] = None
    prompt_text: Optional[str] = None
    source: str = "heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def use_standard_grasp_pipeline(self) -> bool:
        """Return whether recovery should use standard grasp grounding."""
        return self.mode == GRASP

    @property
    def recovery_mode(self) -> str:
        """Return the normalized recovery mode."""
        return self.mode

    @property
    def use_non_prehensile_pipeline(self) -> bool:
        """Return whether recovery should use fingertip grounding."""
        return self.mode == NON_PREHENSILE

    @property
    def should_run_hand_flow(self) -> bool:
        """Return whether the recovery decision requires hand flow."""
        return self.mode == NON_PREHENSILE

    @property
    def recovery_action(self) -> str:
        """Return the recovery action text."""
        return str(self.metadata.get("recovery_action") or self.recovery_prompt or "")

    @property
    def contact_finger(self) -> Optional[str]:
        """Return the selected contact finger, if any."""
        value = self.metadata.get("contact_finger")
        return str(value) if value else None

    def to_paper_dict(self) -> Dict[str, Any]:
        """Serialize the decision using the paper recovery schema."""
        if self.mode == GRASP:
            annotation = {
                "enabled": False,
                "object_name": None,
                "contact_point_definition": None,
                "edit_spec": None,
                "annotated_image_png_base64": None,
            }
            prompt_p = {"enabled": False, "title": None, "text": None}
        else:
            annotation = {
                "enabled": self.annotation_enabled,
                "object_name": self.annotation_object_name,
                "contact_point_definition": self.contact_point_definition,
                "edit_spec": self.annotation_edit_spec or None,
                "annotated_image_png_base64": self.annotated_image_png_base64,
            }
            prompt_p = {
                "enabled": bool(self.prompt_text),
                "title": self.prompt_title,
                "text": self.prompt_text,
            }
        return {
            "recovery_mode": self.mode,
            "recovery_action": self.recovery_action,
            "mode_justification": {
                "discrepancy_summary": self.metadata.get("discrepancy_summary", self.reason),
                "why_this_mode": self.reason,
            },
            "annotation": annotation,
            "prompt_P": prompt_p,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        data = {
            "mode": self.mode,
            "object_prompt": self.object_prompt,
            "contact_prompt": self.contact_prompt,
            "recovery_action": self.recovery_action,
            "recovery_prompt": self.recovery_prompt,
            "contact_point_2d": self.contact_point_2d,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "use_standard_grasp_pipeline": self.use_standard_grasp_pipeline,
            "use_non_prehensile_pipeline": self.use_non_prehensile_pipeline,
            "should_run_hand_flow": self.should_run_hand_flow,
            "metadata": self.metadata,
        }
        data.update(self.to_paper_dict())
        return data


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _canonical_mode(mode: str) -> str:
    value = (mode or "").strip().lower().replace("-", "_")
    if value in {"regrasp", "grasp_recovery", "standard_grasp"}:
        return GRASP
    if value in {"push", "poke", "nudge", "slide", "nonprehensile", "non_prehensile"}:
        return NON_PREHENSILE
    if value not in VALID_MODES:
        raise ValueError(f"Unknown recovery mode: {mode!r}")
    return value


def _normalize_contact_finger(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"^(?:right_|human_right_)", "", normalized)
    normalized = re.sub(r"_(?:finger_)?tip$", "", normalized)
    aliases = {"little": "pinky", "forefinger": "index"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in CANONICAL_CONTACT_FINGERS else None


def _contact_finger_from_prompt(prompt_text: str, data: Dict[str, Any]) -> tuple[Optional[str], str]:
    """Extract the non-prehensile contact finger without extending paper JSON."""

    matches = re.findall(
        r"(?im)^\s*CONTACT\s+FINGER\s*:\s*([a-zA-Z _-]+?)\s*$",
        prompt_text or "",
    )
    if len(matches) != 1:
        raise ValueError(
            "non-prehensile prompt_P must contain exactly one `CONTACT FINGER: <finger>` line"
        )
    parsed = _normalize_contact_finger(matches[0])
    if not parsed:
        raise ValueError(
            f"prompt_P names unsupported contact finger {matches[0]!r}; "
            f"expected one of {CANONICAL_CONTACT_FINGERS}"
        )
    mentioned = {
        finger
        for finger in CANONICAL_CONTACT_FINGERS
        if re.search(rf"\b{re.escape(finger)}\b", prompt_text or "", flags=re.IGNORECASE)
    }
    if mentioned - {parsed}:
        raise ValueError(
            f"prompt_P ambiguously names multiple contact fingers: {sorted(mentioned)}"
        )
    return parsed, "prompt_P"


def _contact_point_from_annotated_png(
    value: Any,
    *,
    original_image_path: Optional[str] = None,
) -> Optional[list[float]]:
    if not value:
        return None
    try:
        raw = base64.b64decode(str(value), validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask = (rgb[..., 0] >= 200) & (rgb[..., 1] <= 80) & (rgb[..., 2] <= 80)
        if original_image_path:
            try:
                with Image.open(original_image_path) as original:
                    original_rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
                if original_rgb.shape == rgb.shape:
                    delta = np.max(
                        np.abs(rgb.astype(np.int16) - original_rgb.astype(np.int16)),
                        axis=-1,
                    )
                    # When the source image is available, only newly drawn red
                    # pixels are a valid annotation.  Falling back to every red
                    # source pixel can silently select an existing red object.
                    mask = mask & (delta >= 40)
            except Exception:
                pass
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            return None
        return [float(np.mean(xs)), float(np.mean(ys))]
    except Exception:
        return None


def _normalized_contact_point(
    data: Dict[str, Any],
    annotation: Dict[str, Any],
    *,
    original_image_path: Optional[str] = None,
) -> Optional[list[float]]:
    value = data.get("contact_point_2d") or annotation.get("contact_point_2d")
    if value is not None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("contact_point_2d must contain exactly [x, y]")
        point = [float(value[0]), float(value[1])]
        if not all(math.isfinite(item) and item >= 0.0 for item in point):
            raise ValueError("contact_point_2d must contain finite nonnegative pixel coordinates")
        return point
    definition = str(annotation.get("contact_point_definition") or data.get("contact_prompt") or "")
    match = re.search(
        r"PIXEL\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        definition,
        flags=re.IGNORECASE,
    )
    if match:
        point = [float(match.group(1)), float(match.group(2))]
        if all(math.isfinite(item) and item >= 0.0 for item in point):
            return point
    return _contact_point_from_annotated_png(
        annotation.get("annotated_image_png_base64"),
        original_image_path=original_image_path,
    )


def _image_to_png_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_solid_red_star(image: Image.Image, contact_point_2d: list[float], radius: int = 12) -> Image.Image:
    """Overlay the Appendix C.3.2 solid red star marker at the contact point."""
    x, y = float(contact_point_2d[0]), float(contact_point_2d[1])
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    points = []
    for i in range(10):
        angle = -90.0 + i * 36.0
        r = radius if i % 2 == 0 else radius * 0.45
        px = x + r * math.cos(math.radians(angle))
        py = y + r * math.sin(math.radians(angle))
        points.append((px, py))
    draw.polygon(points, fill=(255, 0, 0))
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
    return annotated


def annotated_image_base64_from_path(image_path: str, contact_point_2d: list[float]) -> str:
    """Encode an annotated image as base64."""
    with Image.open(image_path) as img:
        return _image_to_png_base64(draw_solid_red_star(img, contact_point_2d))


def recovery_decision_from_model_text(
    text: str,
    context: RecoveryContext,
    *,
    source: str = "vlm",
) -> RecoveryDecision:
    """Parse and validate a recovery decision from VLM output."""
    data = _extract_json_object(text)
    mode = _canonical_mode(str(data.get("recovery_mode") or data.get("mode", "")))
    annotation = data.get("annotation") if isinstance(data.get("annotation"), dict) else {}
    prompt_p = data.get("prompt_P") if isinstance(data.get("prompt_P"), dict) else {}
    justification = data.get("mode_justification") if isinstance(data.get("mode_justification"), dict) else {}

    object_prompt = str(
        data.get("object_prompt")
        or annotation.get("object_name")
        or context.object_prompt
    )
    contact_point = _normalized_contact_point(
        data,
        annotation,
        original_image_path=context.current_image_path,
    )
    reason = str(
        data.get("reason")
        or justification.get("why_this_mode")
        or "model selected recovery mode"
    )
    discrepancy_summary = str(justification.get("discrepancy_summary") or reason)
    # Current responses must explicitly propose the action that will be used
    # throughout recovery. Older ``mode``/``recovery_prompt`` payloads remain
    # readable, but a current ``recovery_mode`` payload must never silently
    # reuse the normal action that just failed.
    raw_recovery_action = data.get("recovery_action")
    if raw_recovery_action is not None and not isinstance(raw_recovery_action, str):
        raise ValueError("recovery response recovery_action must be a string")
    if "recovery_mode" in data and not (raw_recovery_action or "").strip():
        raise ValueError("recovery response requires a non-empty recovery_action")
    legacy_recovery_prompt = data.get("recovery_prompt")
    if legacy_recovery_prompt is not None and not isinstance(legacy_recovery_prompt, str):
        raise ValueError("legacy recovery_prompt must be a string")
    recovery_action = (
        raw_recovery_action
        or legacy_recovery_prompt  # legacy response compatibility
        or ""
    )
    recovery_action = re.sub(
        r"(?im)^\s*CONTACT\s+FINGER\s*:\s*[^\n]+\n?",
        "",
        recovery_action,
    ).strip()
    if not recovery_action:
        raise ValueError("recovery response requires a non-empty recovery_action")

    if mode == NON_PREHENSILE:
        if contact_point is None:
            raise ValueError(
                "non-prehensile recovery requires a usable contact pixel: provide "
                "contact_point_definition ending in `PIXEL: [x, y]` or a red-star annotated PNG"
            )
        if "enabled" in annotation and annotation.get("enabled") is not True:
            raise ValueError("non-prehensile annotation.enabled must be true")
        if "enabled" in prompt_p and prompt_p.get("enabled") is not True:
            raise ValueError("non-prehensile prompt_P.enabled must be true")
        annotation_enabled = True
        annotated_image_png_base64 = annotation.get("annotated_image_png_base64")
        if not annotated_image_png_base64 and contact_point is not None and context.current_image_path:
            annotated_image_png_base64 = annotated_image_base64_from_path(context.current_image_path, contact_point)
        raw_edit_spec = annotation.get("edit_spec") if isinstance(annotation.get("edit_spec"), dict) else {}
        edit_spec = {
            "marker": raw_edit_spec.get("marker", "solid_red_star"),
            "anchor": raw_edit_spec.get("anchor", "center_of_star_is_contact_point"),
            "placement": raw_edit_spec.get("placement")
            or annotation.get("contact_point_definition")
            or data.get("contact_prompt")
            or object_prompt,
        }
        explicit_prompt_text = prompt_p.get("text") or data.get("recovery_prompt")
        if not str(explicit_prompt_text or "").strip():
            raise ValueError(
                "non-prehensile recovery requires an explicit prompt_P.text or legacy "
                "recovery_prompt containing exactly one CONTACT FINGER line"
            )
        prompt_text = str(explicit_prompt_text).strip()
        prompt_title = str(prompt_p.get("title") or "Non-prehensile local recovery prompt")
        contact_finger, finger_source = _contact_finger_from_prompt(prompt_text, data)
        recovery_prompt = prompt_text
        annotation_object_name = annotation.get("object_name") or object_prompt
        contact_point_definition = annotation.get("contact_point_definition") or data.get("contact_prompt")
        contact_prompt = str(data.get("contact_prompt") or contact_point_definition or object_prompt)
    else:
        annotation_enabled = False
        annotated_image_png_base64 = None
        edit_spec = {}
        prompt_text = None
        prompt_title = None
        contact_finger = None
        finger_source = "not_applicable"
        recovery_prompt = ""
        annotation_object_name = None
        contact_point_definition = None
        contact_prompt = ""

    return RecoveryDecision(
        mode=mode,
        object_prompt=object_prompt,
        contact_prompt=contact_prompt,
        recovery_prompt=recovery_prompt,
        contact_point_2d=contact_point,
        annotation_enabled=annotation_enabled,
        annotation_object_name=annotation_object_name,
        contact_point_definition=contact_point_definition,
        annotation_edit_spec=edit_spec,
        annotated_image_png_base64=annotated_image_png_base64,
        prompt_title=prompt_title,
        prompt_text=prompt_text,
        reason=reason,
        confidence=float(data.get("confidence", 0.0)),
        source=source,
        metadata={
            "raw": data,
            "discrepancy_summary": discrepancy_summary,
            "recovery_action": recovery_action,
            "contact_finger": contact_finger,
            "contact_finger_source": finger_source,
            "prompt_contract": prompt_contract("recovery_policy").metadata(),
        },
    )


def heuristic_recovery_decision(context: RecoveryContext) -> RecoveryDecision:
    """Choose a recovery decision using the configured VLM and fallback."""
    text = " ".join(
        [
            context.goal,
            context.current_state,
            context.previous_action,
            context.failure_reason,
            context.object_prompt,
        ]
    ).lower()

    grasp_terms = {
        "grasp",
        "regrasp",
        "pick",
        "lift",
        "carry",
        "hold",
        "drop",
        "dropped",
        "lost grasp",
        "slipped",
        "handle",
        "drawer",
        "door",
        "out of reach",
    }
    non_prehensile_terms = {
        "push",
        "poke",
        "nudge",
        "slide",
        "scoot",
        "tap",
        "tip",
        "rotate",
        "align",
        "small correction",
        "slightly",
        "stuck",
    }

    grasp_score = sum(1 for term in grasp_terms if term in text)
    non_prehensile_score = sum(1 for term in non_prehensile_terms if term in text)

    mode = GRASP
    if non_prehensile_score > grasp_score:
        reason = (
            "text heuristics suggest a small contact correction, but cannot ground a unique "
            "contact pixel and finger; using grasp-mode first-last-frame recovery instead"
        )
    else:
        reason = (
            "heuristic selected grasp recovery because stable object control appears "
            "necessary or the context is ambiguous"
        )

    return RecoveryDecision(
        mode=mode,
        object_prompt=context.object_prompt,
        contact_prompt="",
        recovery_prompt="",
        annotation_enabled=False,
        annotation_object_name=None,
        contact_point_definition=None,
        annotation_edit_spec={},
        prompt_title=None,
        prompt_text=None,
        reason=reason,
        confidence=0.55 if mode == GRASP else 0.6,
        source="heuristic",
        metadata={
            "grasp_score": grasp_score,
            "non_prehensile_score": non_prehensile_score,
            "recovery_action": (
                f"Regrasp the {context.object_prompt} and move it toward the goal: "
                f"{context.goal}"
            ),
            "contact_finger": None,
            "contact_finger_source": "not_applicable",
            "prompt_contract": prompt_contract("recovery_policy").metadata(),
        },
    )


def choose_recovery_strategy(
    context: RecoveryContext,
    *,
    model_text: Optional[str] = None,
    vlm_callable: Optional[Callable[[str], str]] = None,
) -> RecoveryDecision:
    """Choose recovery strategy."""
    if model_text:
        return recovery_decision_from_model_text(model_text, context)
    if vlm_callable is not None:
        response = vlm_callable(context.to_prompt())
        return recovery_decision_from_model_text(response, context)
    return heuristic_recovery_decision(context)


def _load_planner_summary(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_base64_png(path: Path, image_png_base64: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(image_png_base64))


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Choose NovaPlan recovery mode.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--current-state", default="")
    parser.add_argument("--target-state", default="")
    parser.add_argument("--previous-action", default="")
    parser.add_argument("--failure-reason", default="")
    parser.add_argument("--object-prompt", default="object")
    parser.add_argument("--current-image", type=Path, default=None, help="Current observation image to annotate.")
    parser.add_argument("--target-image", type=Path, default=None, help="Previous target image from the selected rollout.")
    parser.add_argument("--planner-summary", type=Path, default=None)
    parser.add_argument("--model-json", default="", help="Optional model JSON response to parse.")
    parser.add_argument("--annotated-image-out", type=Path, default=None, help="Optional path for annotated recovery PNG.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    context = RecoveryContext(
        goal=args.goal,
        current_state=args.current_state,
        target_state=args.target_state,
        previous_action=args.previous_action,
        failure_reason=args.failure_reason,
        object_prompt=args.object_prompt,
        current_image_path=str(args.current_image) if args.current_image else None,
        target_image_path=str(args.target_image) if args.target_image else None,
        planner_summary=_load_planner_summary(args.planner_summary),
    )
    decision = choose_recovery_strategy(context, model_text=args.model_json or None)
    payload = decision.to_dict()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.annotated_image_out is not None and decision.annotated_image_png_base64:
        _write_base64_png(args.annotated_image_out, decision.annotated_image_png_base64)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
