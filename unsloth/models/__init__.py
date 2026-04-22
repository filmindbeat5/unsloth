# Copyright 2023-present, the Unsloth team.
# Licensed under the Apache License, Version 2.0
"""
Unsloth models module.

Provides fast, memory-efficient model loading and patching utilities
for popular LLM architectures using custom CUDA kernels and optimizations.

Note: Also exposes FastVisionModel for multimodal use cases.

Personal note: FastLanguageModel is the main entry point for most use cases.
Use FastVisionModel only when working with vision-language models (e.g. LLaVA).
"""

from .loader import FastLanguageModel
from .loader import FastVisionModel

__all__ = [
    "FastLanguageModel",
    "FastVisionModel",
]
