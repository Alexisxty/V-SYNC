from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .types import InferenceRequest, InferenceResult


class BasePipeline(ABC):
    """所有模型适配器必须实现的统一接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识（用于日志/结果归档）。"""

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """对单个样本进行推理。"""

    def batch_predict(self, requests: Iterable[InferenceRequest]) -> list[InferenceResult]:
        """默认批量推理（可被子类重写）。"""
        return [self.predict(req) for req in requests]
