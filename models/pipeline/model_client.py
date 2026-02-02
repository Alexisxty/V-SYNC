from __future__ import annotations

from typing import Protocol

from .types import InferenceRequest, InferenceResult


class ModelClient(Protocol):
    """Pipeline 统一调用的模型客户端接口。"""

    @property
    def model_name(self) -> str:
        ...

    def predict(self, request: InferenceRequest) -> InferenceResult:
        ...
