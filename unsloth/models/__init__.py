# Copyright 2023-present, the Unsloth team.
# Licensed under the Apache License, Version 2.0
"""
Unsloth models module.

Provides fast, memory-efficient model loading and patching utilities
for popular LLM architectures using custom CUDA kernels and optimizations.
"""

from .loader import FastLanguageModel

__all__ = [
    "FastLanguageModel",
]
