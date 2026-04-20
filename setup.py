from setuptools import setup, find_packages
import os

# Read the README for the long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="unsloth",
    version="2024.1.0",
    description="2x faster, 60% less memory LLM finetuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Unsloth AI",
    author_email="danielhanchen@gmail.com",
    url="https://github.com/unslothai/unsloth",
    license="Apache 2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.16.0",
        "sentencepiece>=0.1.99",
        "tqdm",
        "psutil",
        "wheel>=0.42.0",
        "packaging>=23.1",
        "numpy",
        "accelerate>=0.26.0",
        "peft>=0.7.1",
        "bitsandbytes>=0.41.3",
        # Relaxed protobuf constraint to allow v4.x which fixes compatibility
        # with newer grpc-based tools I use locally
        "protobuf>=3.20.0",
        "huggingface_hub",
        "hf_transfer",
        "trl>=0.7.9",
        "xformers",
        "triton",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "colab": [
            "ipython",
            "ipywidgets",
        ],
        # Personal extra: tools I find useful when experimenting locally
        "local": [
            "ipython",
            "ipywidgets",
            "matplotlib",
            "wandb",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "llm",
        "finetuning",
        "lora",
        "qlora",
        "transformers",
        "mistral",
        "llama",
        "efficient training",
    ],
    entry_points={
        "console_scripts": [
            "unsloth=unsloth.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
