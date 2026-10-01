# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in model transform ordering and conflict relations."""

from typing import TypeAlias

from .async_tensor_parallel import AsyncTensorParallelTransform
from .base import ModelConfigTransform
from .context_parallel import ContextParallelTransform
from .lora import LoRATransform

TransformType: TypeAlias = type[ModelConfigTransform]
PrecedenceRelation: TypeAlias = tuple[TransformType, TransformType]
ConflictRelation: TypeAlias = tuple[TransformType, TransformType]

__all__ = ["CONFLICTS", "PRECEDES"]


# Each pair is (prerequisite, dependent).
PRECEDES: tuple[PrecedenceRelation, ...] = (
    # LoRA freezes configs by creating dynamic subclasses. Context parallelism
    # must replace attention configs first because convert_config_type requires
    # its replacement config to inherit the current config type.
    (ContextParallelTransform, LoRATransform),
)


# Each pair is unordered.
CONFLICTS: tuple[ConflictRelation, ...] = (
    # Async kernels invoke fused autograd functions directly instead of the
    # projection's _linear method, which would silently omit LoRA computation.
    (AsyncTensorParallelTransform, LoRATransform),
    # LoRA freezes every non-target config. Applying it more than once would
    # make freezing and adapter configuration depend on transform order.
    (LoRATransform, LoRATransform),
)
