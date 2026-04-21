# Copyright 2023-present, the Unsloth authors.
# Licensed under the Apache License, Version 2.0
#
# This file is the main entry point for the unsloth package.

__version__ = "2024.1"
__author__ = "Unsloth Authors"
__license__ = "Apache 2.0"

import sys
import os

# Ensure we are running on Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"Unsloth requires Python 3.8 or higher. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Check for CUDA availability early
def _check_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUDA is not available. Unsloth requires a CUDA-capable GPU for optimal performance. "
                "Some features may not work correctly.",
                UserWarning,
                stacklevel=3,
            )
        return torch.cuda.is_available()
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it via: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

_cuda_available = _check_cuda()

# Core imports exposed at package level
try:
    from unsloth.models import FastLanguageModel
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import FastLanguageModel: {e}. "
        "Some dependencies may be missing. Run: pip install unsloth[full]",
        ImportWarning,
        stacklevel=2,
    )
    FastLanguageModel = None

try:
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
except ImportError:
    UnslothTrainer = None
    UnslothTrainingArguments = None

__all__ = [
    "FastLanguageModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "__version__",
]


def get_version():
    """Return the current version of unsloth."""
    return __version__


def is_cuda_available():
    """Return whether CUDA is available on the current system."""
    return _cuda_available


def print_info():
    """Print system and package information useful for debugging."""
    import torch
    print(f"Unsloth version   : {__version__}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version      : {torch.version.cuda}")
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory (GB)   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    try:
        import transformers
        print(f"Transformers ver  : {transformers.__version__}")
    except ImportError:
        print("Transformers      : not installed")
    try:
        import peft
        print(f"PEFT version      : {peft.__version__}")
    except ImportError:
        print("PEFT              : not installed")
    try:
        import trl
        print(f"TRL version       : {trl.__version__}")
    except ImportError:
        print("TRL               : not installed")
