from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    AVT = "audio+vision+text"
    VT = "vision+text"
    AT = "audio+text"
