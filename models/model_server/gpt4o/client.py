from __future__ import annotations

from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.openai_compat_tester import OpenAICompatTester
from config.settings import CONFIG


class GPT4oClient:
    @property
    def model_name(self) -> str:
        return "gpt4o"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        asr_content = request.metadata.get("asr_content") if request.metadata else None
        options = request.options or []
        use_video = True
        if request.metadata:
            use_video = bool(request.metadata.get("use_video", True))
        include_audio = False
        prompt_parts = []
        if user_prompt:
            prompt_parts.append(user_prompt)
        if asr_content:
            prompt_parts.append(f"[ASR]\n{asr_content}")
        if options:
            prompt_parts.append("Options:\n" + "\n".join(options))
        prompt_parts.append(CONFIG.prompt("answer_format"))
        user_prompt = "\n\n".join(prompt_parts) if prompt_parts else None
        include_audio = False

        model_config = CONFIG.model("gpt4o")
        model_name = model_config.get("model_name", "gpt-4o")
        tester = OpenAICompatTester(model_name=model_name)
        raw_answer = tester.call(
            request.video_path,
            request.question,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=use_video,
        )
        return InferenceResult(answer=raw_answer or "", raw_response=raw_answer)
