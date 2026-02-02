from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.paths import PATHS
from config.settings import CONFIG
from models.pipeline.model_client import ModelClient
from models.pipeline.types import InferenceRequest, InferenceResult



@dataclass(frozen=True)
class Level1Config:
    dataset_path: Path
    video_dir: Path
    output_path: Path
    log_dir: Path
    max_samples: Optional[int] = None
    start_index: int = 0
    resume: bool = False


class Level1Pipeline:
    """Level1 统一评测流程：构建问题 → 调用模型 → 评测 → 输出结果。"""

    def __init__(self, omni_test: ModelClient, config: Level1Config) -> None:
        self.omni_test = omni_test
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        model_log_dir = self.config.log_dir / self.omni_test.model_name
        model_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = model_log_dir / f"level1_{self.omni_test.model_name}_{timestamp}.log"

        logger = logging.getLogger(f"level1_{self.omni_test.model_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_dataset(self) -> List[dict]:
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _build_request(self, sample: dict) -> InferenceRequest:
        options = sample.get("options") or []
        system_prompt = CONFIG.benchmark("level1.system_prompt", "").strip()
        user_prompt_base = CONFIG.benchmark("level1.user_prompt", "").strip()
        answer_format = CONFIG.prompt("answer_format").strip()
        use_video, use_audio = self._resolve_modality()
        asr_content = sample.get("asr_content") or ""
        if not use_audio:
            asr_content = ""

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"[SYSTEM]\n{system_prompt}")
        if asr_content:
            prompt_parts.append(f"[ASR]\n{asr_content}")
        if options:
            prompt_parts.append("Options:\n" + "\n".join(options))
        if user_prompt_base:
            prompt_parts.append(user_prompt_base)
        if answer_format:
            prompt_parts.append(answer_format)

        user_prompt = "\n\n".join(prompt_parts) if prompt_parts else None
        return InferenceRequest(
            video_path=str(self.config.video_dir / sample["video_path"]),
            question=sample["question"],
            options=options,
            metadata={
                "correct_answer": sample.get("correct_answer"),
                "sample_id": sample.get("id"),
                "asr_content": asr_content,
                "user_prompt": user_prompt,
                "use_video": use_video,
                "use_audio": use_audio,
            },
        )

    def _resolve_modality(self) -> tuple[bool, bool]:
        raw = str(CONFIG.benchmark("level1.modality", "avt")).strip().lower()
        if raw in {"vt", "v", "vision", "vision+text", "video+text"}:
            return True, False
        if raw in {"at", "a", "audio", "audio+text"}:
            return False, True
        return True, True

    def _normalize_answer(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return ""
        if answer[0].upper() in {"A", "B", "C", "D"}:
            return answer[0].upper()
        match = re.search(r"\b([A-D])\b", answer.upper())
        if match:
            return match.group(1)
        return answer

    def _score(self, prediction: str, correct: Optional[str]) -> bool:
        if not correct:
            return False
        return prediction == correct.strip().upper()

    def run(self) -> dict:
        dataset = self.load_dataset()
        if self.config.start_index > 0:
            dataset = dataset[self.config.start_index :]
        if self.config.max_samples is not None:
            dataset = dataset[: self.config.max_samples]

        results = []
        correct = 0
        total = 0

        processed_ids = set()
        if self.config.resume and self.config.output_path.exists():
            try:
                with open(self.config.output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                results = existing.get("results", [])
                processed_ids = {r.get("id") for r in results if r.get("id") is not None}
                correct = sum(1 for r in results if r.get("is_correct"))
                total = len(results)
                self.logger.info("Resume enabled: %s processed, %s correct", total, correct)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load existing results: %s", exc)

        for sample in dataset:
            if processed_ids and sample.get("id") in processed_ids:
                continue
            request = self._build_request(sample)
            result = self.omni_test.predict(request)
            prediction = self._normalize_answer(result.answer)
            is_correct = self._score(prediction, sample.get("correct_answer"))
            total += 1
            if is_correct:
                correct += 1

            results.append(
                {
                    "id": sample.get("id"),
                    "video_path": sample.get("video_path"),
                    "question": sample.get("question"),
                    "options": sample.get("options"),
                    "correct_answer": sample.get("correct_answer"),
                    "prediction": prediction,
                    "raw_response": result.raw_response,
                    "is_correct": is_correct,
                }
            )

            self.logger.info(
                "[%s] prediction=%s correct=%s",
                sample.get("id"),
                prediction,
                sample.get("correct_answer"),
            )

        accuracy = (correct / total * 100) if total > 0 else 0.0
        payload = {
            "model": self.omni_test.model_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.logger.info("Accuracy %.2f%% (%s/%s)", accuracy, correct, total)
        self.logger.info("Results saved: %s", self.config.output_path)
        return payload


def default_level1_config(model_name: str) -> Level1Config:
    dataset_path = CONFIG.benchmark("level1.dataset_path", "")
    video_dir = CONFIG.benchmark("level1.video_dir", "")
    output_dir = CONFIG.benchmark("level1.output_dir", "")
    output_pattern = CONFIG.benchmark("level1.output_pattern", "results_{model}_level1.json")
    log_dir = CONFIG.benchmark("level1.log_dir", "")

    dataset_path = Path(dataset_path) if dataset_path else PATHS.data_level_1 / "dataset.json"
    video_dir = Path(video_dir) if video_dir else PATHS.data_level_1 / "videos"
    output_base = Path(output_dir) if output_dir else PATHS.results_dir
    output_path = output_base / output_pattern.format(model=model_name)
    log_dir = Path(log_dir) if log_dir else PATHS.results_logs

    return Level1Config(
        dataset_path=dataset_path,
        video_dir=video_dir,
        output_path=output_path,
        log_dir=log_dir,
        resume=False,
    )


def run_level1(omni_test: ModelClient) -> dict:
    pipeline = Level1Pipeline(omni_test, default_level1_config(omni_test.model_name))
    return pipeline.run()
