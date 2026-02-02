from __future__ import annotations

import base64
import logging
import os
from typing import List, Optional

import requests

from config.settings import CONFIG
from config.paths import PATHS


class OpenAICompatTester:
    """OpenAI 兼容多模态调用（文本 + 图像）。"""

    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model_name = model_name
        self.base_url = api_base or os.getenv("OPENAI_API_BASE") or CONFIG.api("openai").get("base_url")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or CONFIG.api("openai").get("api_key")
        if not self.base_url or not self.api_key:
            raise ValueError("Missing OpenAI API base_url or api_key. Check .env or config/config.yaml.")
        self.logger = logging.getLogger(f"openai_compat_tester.{model_name}")
        if not self.logger.handlers:
            model_log_dir = PATHS.results_logs / model_name
            model_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = model_log_dir / "openai_compat_tester.log"
            handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _extract_frames(self, video_path: str, frame_interval_sec: int, max_frames: Optional[int]) -> List[str]:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            self.logger.error("OpenCV is not available: %s", exc)
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(round(fps * frame_interval_sec)) if fps and fps > 0 else 0

        frames_base64: List[str] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if step and idx % step != 0:
                idx += 1
                continue

            frame = cv2.resize(frame, (128, 128))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frames_base64.append(base64.b64encode(buffer).decode('utf-8'))

            if max_frames is not None and len(frames_base64) >= max_frames:
                break

            if step == 0:
                break
            idx += 1

        cap.release()
        return frames_base64

    def call(
        self,
        video_path: str,
        question: str,
        user_prompt: Optional[str] = None,
        model_params: Optional[dict] = None,
        include_images: bool = True,
    ) -> str:
        frame_interval_sec = int(CONFIG.runtime("frame_interval_sec", 1))
        max_frames = CONFIG.runtime("max_frames", None)
        if max_frames is not None:
            try:
                max_frames = int(max_frames)
            except (TypeError, ValueError):
                max_frames = None

        frames: List[str] = []
        if include_images:
            frames = self._extract_frames(video_path, frame_interval_sec, max_frames)
            if not frames:
                return ""

        full_question = f"{user_prompt}\n\n{question}" if user_prompt else question
        content = [
            {"type": "text", "text": full_question},
        ]
        if include_images:
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        if model_params:
            max_tokens = model_params.get("max_tokens")
            temperature = model_params.get("temperature")
            top_p = model_params.get("top_p")
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if top_p is not None:
                payload["top_p"] = float(top_p)

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
