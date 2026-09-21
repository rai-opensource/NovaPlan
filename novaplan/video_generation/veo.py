# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Generate video rollouts with Google Veo through Vertex AI."""

import os
import time
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# The new unified SDK that supports Vertex AI correctly
from google import genai
from google.genai import types


DEFAULT_VEO_MODEL = "veo-3.1-generate-001"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VEO_OPERATION_TIMEOUT_SECONDS = 1800.0
DEFAULT_VEO_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_VEO_DOWNLOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_VEO_POLL_INTERVAL_SECONDS = 5.0


def _positive_seconds(
    explicit: Optional[float],
    *,
    env_name: str,
    default: float,
) -> float:
    raw_value = explicit if explicit is not None else os.getenv(env_name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive number of seconds") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive number of seconds")
    return value


class VeoGenerationBatch(list[np.ndarray]):
    """List-compatible Veo output with original sample indices and failures."""

    def __init__(
        self,
        videos: List[np.ndarray],
        *,
        sample_indices: List[int],
        sample_errors: Dict[int, str],
    ):
        super().__init__(videos)
        self.sample_indices = tuple(int(index) for index in sample_indices)
        self.sample_errors = dict(sample_errors)


def overlay_text(frame: np.ndarray, text: str, pos: str = "bottom", pad: int = 8) -> np.ndarray:
    """Overlay text on a frame image."""
    H, W, _ = frame.shape
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        # Try a common Linux font, fallback to default
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    text = text or ""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x0, y0 = (pad, H - th - 2*pad) if pos == "bottom" else (pad, pad)

    draw.rectangle([x0-pad, y0-pad, x0+tw+pad, y0+th+pad], fill=(255, 255, 255))
    draw.text((x0, y0), text, fill=(0, 0, 0), font=font)
    return np.array(img, dtype=np.uint8)

class VeoVideoGenerationClient:
    """
    Veo 3.1 Adapter for Vertex AI.

    Environment Variables Required:
      - GOOGLE_CLOUD_PROJECT: Your Google Cloud Project ID (e.g., 'my-genai-project')
      - Google Application Default Credentials or equivalent Vertex AI auth.

    Optional endpoint overrides:
      - GOOGLE_CLOUD_LOCATION: Vertex location; defaults to ``global``.
      - VEO3_MODEL: Veo publisher model; defaults to ``veo-3.1-generate-001``.
    """

    def __init__(
        self,
        mock: bool = False,
        seed: int = 0,
        debug_dir: Optional[Path] = None,
        operation_timeout_s: Optional[float] = None,
        request_timeout_s: Optional[float] = None,
        download_timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
    ):
        self.mock = mock
        self.rng = np.random.default_rng(seed)
        self.debug_dir = debug_dir
        self.operation_timeout_s = _positive_seconds(
            operation_timeout_s,
            env_name="VEO_OPERATION_TIMEOUT_SECONDS",
            default=DEFAULT_VEO_OPERATION_TIMEOUT_SECONDS,
        )
        self.request_timeout_s = _positive_seconds(
            request_timeout_s,
            env_name="VEO_REQUEST_TIMEOUT_SECONDS",
            default=DEFAULT_VEO_REQUEST_TIMEOUT_SECONDS,
        )
        self.download_timeout_s = _positive_seconds(
            download_timeout_s,
            env_name="VEO_DOWNLOAD_TIMEOUT_SECONDS",
            default=DEFAULT_VEO_DOWNLOAD_TIMEOUT_SECONDS,
        )
        self.poll_interval_s = _positive_seconds(
            poll_interval_s,
            env_name="VEO_POLL_INTERVAL_SECONDS",
            default=DEFAULT_VEO_POLL_INTERVAL_SECONDS,
        )

        # Vertex AI retired the preview endpoint in favor of the GA -001 model.
        self._model = os.getenv("VEO3_MODEL", DEFAULT_VEO_MODEL)

        # Vertex AI requires a Project ID and Location
        self._project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_VERTEX_LOCATION)

        if not self.mock:
            if not self._project_id:
                raise ValueError(
                    "Missing GOOGLE_CLOUD_PROJECT environment variable. "
                    "This is required for Vertex AI execution."
                )

            print(f"Initializing Veo Adapter on Vertex AI ({self._location}) for model: {self._model}")

            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location,
                http_options=types.HttpOptions(
                    timeout=max(1, int(self.request_timeout_s * 1000))
                ),
            )

    @staticmethod
    def _np_to_png_bytes(arr: np.ndarray) -> bytes:
        import io
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _mp4_to_numpy(mp4_bytes: bytes) -> np.ndarray:
        """Convert MP4 bytes to numpy array (T, H, W, 3) RGB."""
        import os
        import imageio.v3 as iio

        # Write bytes to temp file so ffmpeg can read it
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(mp4_bytes)
            tmp_path = tmp.name

        try:
            # ATTEMPT 1: Auto-detect (Safest for modern imageio)
            # We remove 'plugin="ffmpeg"' so imageio finds the best available backend (pyav or ffmpeg)
            try:
                vid = iio.imread(tmp_path, index=None)
            except Exception:
                # ATTEMPT 2: Fallback to V2 legacy reader (often more stable on older setups)
                import imageio.v2 as iio2
                vid = iio2.mimread(tmp_path, memtest=False)
                vid = np.array(vid)

            # Ensure correct shape (T, H, W, C)
            if vid.ndim == 3: # Handle single frame edge case
                vid = vid[np.newaxis, ...]

            if vid.dtype != np.uint8:
                vid = vid.astype(np.uint8)

            return vid

        except Exception as e:
            raise RuntimeError(
                f"Failed to decode MP4 video: {e}. "
                "Try installing the local-planning environment: `pixi install -e local-planning`"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _resample_video_frames(video: np.ndarray, target_frames: int) -> np.ndarray:
        """Uniformly resample a rollout to the shared planner frame count."""

        video = np.asarray(video)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must be (T, H, W, 3)")
        if target_frames < 1:
            raise ValueError("target_frames must be positive")
        source_frames = int(video.shape[0])
        if source_frames < 1:
            raise ValueError("video must contain at least one frame")
        if source_frames == target_frames:
            return video
        indices = np.rint(
            np.linspace(0, source_frames - 1, num=target_frames)
        ).astype(np.int64)
        return video[indices]

    def _wait_for_operation(self, operation, *, sample_index: int):
        """Poll one Vertex operation until completion or the configured deadline."""

        deadline = time.monotonic() + self.operation_timeout_s
        while not operation.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Veo sample {sample_index + 1} timed out after "
                    f"{self.operation_timeout_s:g} seconds"
                )
            time.sleep(min(self.poll_interval_s, remaining))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Veo sample {sample_index + 1} timed out after "
                    f"{self.operation_timeout_s:g} seconds"
                )
            operation = self._client.operations.get(operation)
        return operation

    def _download_video_bytes(self, uri: str) -> bytes:
        """Download an HTTP video result with a bounded request."""

        response = requests.get(uri, timeout=self.download_timeout_s)
        response.raise_for_status()
        return response.content

    def generate_rollouts(self, start_frame: np.ndarray, action_text: str,
                          num_samples: int = 1, num_frames: int = 41,
                          fps: int = 24, jitter: int = 0,
                          last_frame: Optional[np.ndarray] = None,
                          size: Optional[str] = None,
                          seed: Optional[int] = None,
                          enable_flow: bool = False,
                          mask_prompt: Optional[str] = None,
                          negative_prompt: Optional[str] = None,
                          on_video: Optional[Callable[[int, np.ndarray], None]] = None,
                          max_workers: Optional[int] = None,
                          **kwargs) -> List[np.ndarray]:
        # These are shared video-client compatibility fields. Hybrid mode uses
        # ``on_video`` to queue external flow; the Veo adapter itself only
        # generates video.
        """Generate and return video rollout candidates."""
        del size, enable_flow, mask_prompt, kwargs
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        start_frame = np.asarray(start_frame, dtype=np.uint8)
        if start_frame.ndim != 3 or start_frame.shape[-1] != 3:
            raise ValueError("start_frame must be (H, W, 3) RGB uint8")
        if last_frame is not None:
            last_frame = np.asarray(last_frame, dtype=np.uint8)
            if last_frame.ndim != 3 or last_frame.shape[-1] != 3:
                raise ValueError("last_frame must be (H, W, 3) RGB uint8")
            if last_frame.shape != start_frame.shape:
                last_frame = np.asarray(
                    Image.fromarray(last_frame).resize(
                        (start_frame.shape[1], start_frame.shape[0]),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.uint8,
                )
        if self.mock:
            H, W, _ = start_frame.shape
            outs = []
            print(f"[Mock] Generating {num_samples} rollout(s)...")
            for i in range(num_samples):
                frames = []
                # Use jitter for mock randomness if provided
                dx = self.rng.integers(-jitter, jitter + 1) if jitter > 0 else 0
                for t in range(num_frames):
                    alpha = t / max(1, num_frames - 1)
                    if last_frame is not None:
                        canvas = np.rint(
                            (1.0 - alpha) * start_frame.astype(np.float32)
                            + alpha * last_frame.astype(np.float32)
                        ).astype(np.uint8)
                    else:
                        sx = int(dx * alpha)
                        canvas = np.full((H, W, 3), 245, dtype=np.uint8)

                        src_x_start = max(0, -sx)
                        src_x_end = min(W, W - sx)
                        dst_x_start = max(0, sx)
                        dst_x_end = min(W, W + sx)

                        if dst_x_end > dst_x_start:
                            canvas[:, dst_x_start:dst_x_end] = start_frame[:, src_x_start:src_x_end]

                    # Preserve exact conditioning frames in mock mode so tests can
                    # assert the first/last-frame contract.
                    if t not in {0, num_frames - 1}:
                        canvas = overlay_text(canvas, action_text, pos="top")
                    frames.append(canvas)
                video = np.stack(frames, axis=0)
                if on_video is not None:
                    on_video(i, video)
                outs.append(video)
            return outs

        print(f"Generating {num_samples} Veo samples via Vertex AI...")
        image_bytes = self._np_to_png_bytes(start_frame)
        last_image = None
        if last_frame is not None:
            last_image = types.Image(
                image_bytes=self._np_to_png_bytes(last_frame),
                mime_type="image/png",
            )

        def gen_one(i: int) -> np.ndarray:
            t0 = time.time()
            try:
                # Veo determines the native frame rate. The client resamples the
                # result locally instead of passing an unsupported API field.

                config_kwargs = {
                    "aspect_ratio": "16:9",
                    "duration_seconds": 4,
                }
                if last_image is not None:
                    # google-genai exposes FLF conditioning on the generation
                    # config rather than as a second top-level image argument.
                    config_kwargs["last_frame"] = last_image
                if negative_prompt:
                    config_kwargs["negative_prompt"] = str(negative_prompt)
                if seed is not None:
                    config_kwargs["seed"] = int(seed) + i

                operation = self._client.models.generate_videos(
                    model=self._model,
                    prompt=action_text,
                    image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
                    config=types.GenerateVideosConfig(**config_kwargs),
                )

                print(f"Sample {i+1}: Operation submitted. Polling...")
                operation = self._wait_for_operation(operation, sample_index=i)

                if not operation.result or not operation.result.generated_videos:
                    raise RuntimeError(f"API finished but returned no video. Status: {operation.status}")

                video_obj = operation.result.generated_videos[0].video

                if video_obj.video_bytes:
                    mp4_bytes = video_obj.video_bytes
                elif video_obj.uri:
                    if video_obj.uri.startswith("http"):
                        mp4_bytes = self._download_video_bytes(video_obj.uri)
                    else:
                        raise RuntimeError(f"Video saved to {video_obj.uri}. Storage client required.")
                else:
                    raise RuntimeError("Response contained neither video_bytes nor a URI.")

                print(f"Veo sample {i+1} completed in {time.time()-t0:.2f}s")
                vid = self._mp4_to_numpy(mp4_bytes)
                native_frames = int(vid.shape[0])
                vid = self._resample_video_frames(vid, num_frames)
                if native_frames != num_frames:
                    print(
                        f"Veo sample {i+1}: temporally resampled "
                        f"{native_frames} -> {num_frames} frames"
                    )

                return vid

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Quota" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    print(f"!!! QUOTA HIT on Sample {i+1} !!!")
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    err_msg += (
                        f" Configured Vertex model={self._model!r}, "
                        f"location={self._location!r}. Use "
                        "VEO3_MODEL=veo-3.1-generate-001 and "
                        "GOOGLE_CLOUD_LOCATION=global, or verify that the project "
                        "has Vertex AI access to the configured model."
                    )
                raise RuntimeError(f"Veo generation failed: {err_msg}") from e

        worker_limit = max_workers
        if worker_limit is None:
            worker_limit = int(os.getenv("VEO_MAX_PARALLEL", os.getenv("VEO_MAX_WORKERS", "8")))
        worker_limit = max(1, min(num_samples, worker_limit))
        print(f"Submitting {num_samples} Veo samples with max_parallel={worker_limit}")

        outs_by_index: List[Optional[np.ndarray]] = [None] * num_samples
        sample_errors: Dict[int, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as ex:
            futs = {ex.submit(gen_one, i): i for i in range(num_samples)}
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                try:
                    video = fut.result()
                except Exception as exc:
                    sample_errors[idx] = str(exc)
                    print(f"Veo sample {idx + 1} failed: {exc}")
                    continue
                outs_by_index[idx] = video
                if on_video is not None:
                    on_video(idx, video)

        sample_indices = [
            index for index, video in enumerate(outs_by_index) if video is not None
        ]
        videos = [
            video for video in outs_by_index if video is not None
        ]
        if not videos and sample_errors:
            details = "; ".join(
                f"sample {index + 1}: {error}"
                for index, error in sorted(sample_errors.items())
            )
            raise RuntimeError(f"All {num_samples} Veo samples failed: {details}")
        if sample_errors:
            print(
                f"Veo batch completed with {len(videos)}/{num_samples} successful "
                f"sample(s); {len(sample_errors)} failed."
            )
        return VeoGenerationBatch(
            videos,
            sample_indices=sample_indices,
            sample_errors=sample_errors,
        )


# Backward-compatible import for existing integrations. New code should import
# VeoVideoGenerationClient from novaplan.video_generation.
Veo3VideoAdapter = VeoVideoGenerationClient
