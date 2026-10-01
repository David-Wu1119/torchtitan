# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ordering and application of model transforms."""

import copy
from typing import cast, Protocol, TypeVar

from torchtitan.config.configs import TrainingConfig

from torchtitan.config.configurable import Configurable
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.protocols.module import Module

from . import relations
from .base import ModelConfigTransform, ModelConfigTransformContext

__all__ = ["apply_transforms", "transform_model_config_"]


class _TransformableConfig(Protocol):
    model: Module.Config
    training: TrainingConfig
    parallelism: ParallelismConfig

    def __post_init__(self) -> None:
        ...


_ConfigT = TypeVar("_ConfigT", bound=Configurable.Config)


def _runs_after(
    candidate: ModelConfigTransform,
    other: ModelConfigTransform,
) -> bool:
    if isinstance(other, candidate.run_after):
        return True
    return any(
        isinstance(candidate, dependent) and isinstance(other, prerequisite)
        for prerequisite, dependent in relations.PRECEDES
    )


def _ordered(
    transforms: list[ModelConfigTransform],
) -> list[ModelConfigTransform]:
    """Stable-sort transforms by their ``run_after`` declarations."""

    # Not the best performance but simple enough. Given that there are
    # not many transforms, this is acceptable. We can improve it later.
    ordered: list[ModelConfigTransform] = []
    remaining = list(transforms)
    while remaining:
        for i, candidate in enumerate(remaining):
            blockers = [
                other
                for other in remaining
                if other is not candidate and _runs_after(candidate, other)
            ]
            if not blockers:
                ordered.append(remaining.pop(i))
                break
        else:
            unresolved = ", ".join(type(t).__qualname__ for t in remaining)
            raise ValueError(f"Could not order unresolved transforms: {unresolved}.")
    return ordered


def _conflicts(
    transform: ModelConfigTransform,
    other: ModelConfigTransform,
) -> bool:
    if any(
        isinstance(other, conflicting_type)
        for conflicting_type in transform.conflicts_with
    ) or any(
        isinstance(transform, conflicting_type)
        for conflicting_type in other.conflicts_with
    ):
        return True
    return any(
        (isinstance(transform, left) and isinstance(other, right))
        or (isinstance(transform, right) and isinstance(other, left))
        for left, right in relations.CONFLICTS
    )


def _reject_conflicts(transforms: list[ModelConfigTransform]) -> None:
    for i, transform in enumerate(transforms):
        for other in transforms[i + 1 :]:
            if _conflicts(transform, other):
                raise ValueError(
                    f"{type(transform).__qualname__} and "
                    f"{type(other).__qualname__} cannot be combined."
                )


def transform_model_config_(
    model: Module.Config,
    transforms: list[ModelConfigTransform],
    *,
    context: ModelConfigTransformContext,
) -> Module.Config:
    """Apply every transform to ``model`` and return the rewritten root.

    Rewrites in place, so copy ``model`` first to keep the original. Validation
    is the caller's job.
    """
    _reject_conflicts(transforms)
    for transform in _ordered(transforms):
        model = transform.transform(model, context=context)
    return model


def apply_transforms(
    config: _ConfigT,
    transforms: list[ModelConfigTransform],
    *,
    context: ModelConfigTransformContext | None = None,
) -> _ConfigT:
    """Apply every transform to a copy of ``config`` and return it.

    Set all training options before calling this function. It orders the
    transforms, applies them, and validates the result.
    """
    working = copy.deepcopy(config)
    transformable = cast(_TransformableConfig, working)
    if context is None:
        context = ModelConfigTransformContext(
            training=transformable.training,
            parallelism=transformable.parallelism,
        )
    transformable.model = transform_model_config_(
        transformable.model,
        transforms,
        context=context,
    )
    transformable.__post_init__()
    return working
