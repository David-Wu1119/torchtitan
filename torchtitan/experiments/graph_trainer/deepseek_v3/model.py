# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.deepseek_v3 import DeepSeekV3Model

from ..model import GraphTrainerModel


class GraphTrainerDeepSeekV3Model(GraphTrainerModel, DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        enable_autoparallel: bool = False
        """Shard with the AutoParallel solver instead of the manual sharding plan."""

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, skip_dp: bool = False, **kwargs):
        if self.config.enable_autoparallel:
            if skip_dp:
                raise ValueError("GraphTrainer models do not support skip_dp=True.")
            from .parallelize_autoparallel import parallelize_autoparallel_deepseekv3

            return parallelize_autoparallel_deepseekv3(self, **kwargs)
        return super().parallelize(skip_dp=skip_dp, **kwargs)
