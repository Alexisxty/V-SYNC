from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InferenceRequest:
    """统一推理请求结构。"""

    video_path: str
    question: str
    options: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    """统一推理结果结构。"""

    answer: str
    raw_response: Optional[str] = None
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
