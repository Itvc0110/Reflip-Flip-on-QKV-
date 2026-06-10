"""Runtime defaults for Hugging Face text-only experiments.

This repo only uses PyTorch causal language models. Some notebook images ship
with partially installed TensorFlow/JAX/TorchVision stacks; Transformers may
probe those optional backends during import and fail before loading Llama. Set
these flags before importing `transformers` in model-loading scripts.
"""

from __future__ import annotations

import os


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
