# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import dist_moe
import pytest
import torch
from torch.distributed.pipelining import PipelineStageInfo

from torchtitan.config import transform as transform_api
from torchtitan.config.transform import (
    apply_transforms,
    DistMoeTransform,
    GroupedLinearLoRAHandler,
    LinearLoRAHandler,
    LoRATransform,
    MXFP8DistMoeTransform,
)
from torchtitan.models.common.activation import SiTUGLU, SwiGLU
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.quantization.mxfp8 import MXFP8DistMoeRoutedExperts
from torchtitan.training_engine import TrainingEngine


def _parameter_initializers() -> dict[str, Any]:
    return {
        "w1_EFD": torch.nn.init.zeros_,
        "w2_EDF": torch.nn.init.zeros_,
        "w3_EFD": torch.nn.init.zeros_,
    }


def _stock_config(*, dim: int = 32) -> RoutedExperts.Config:
    return make_routed_experts_config(
        dim=dim,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init=_parameter_initializers(),
        comm_backend="standard",
    )


def _get_lora_dist_moe_routed_experts():
    from torchtitan.models.common.lora import get_lora_dist_moe_routed_experts

    return get_lora_dist_moe_routed_experts


def _get_dist_moe_lora_handler():
    handler_cls = getattr(transform_api, "DistMoeLoRAHandler", None)
    assert handler_cls is not None, "DistMoeLoRAHandler is not exported"
    return handler_cls


def _lora_dist_moe_config(
    *,
    output_postprocess: Module.Config | None = None,
) -> DistMoeRoutedExperts.Config:
    stock = _stock_config()
    stock.output_postprocess = output_postprocess
    base_config = cast(DistMoeRoutedExperts.Config, DistMoeTransform().transform(stock))
    lora_cls = _get_lora_dist_moe_routed_experts()(DistMoeRoutedExperts)
    return lora_cls.Config(
        param_init=base_config.param_init,
        sharding_config=base_config.sharding_config,
        w13=base_config.w13,
        w2=base_config.w2,
        token_dispatcher=base_config.token_dispatcher,
        activation_fn=base_config.activation_fn,
        output_postprocess=base_config.output_postprocess,
        inplace_wgrad_accum=base_config.inplace_wgrad_accum,
        bf16_grouped_gemm_preset=base_config.bf16_grouped_gemm_preset,
        rank=8,
        alpha=16.0,
    )


def _build_lora_dist_moe(
    *,
    output_postprocess: Module.Config | None = None,
) -> DistMoeRoutedExperts:
    module = _lora_dist_moe_config(output_postprocess=output_postprocess).build()
    module.init_states()
    return cast(DistMoeRoutedExperts, module)


def _fill_lora_dist_moe_operands(module: DistMoeRoutedExperts) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(values / (10 * index))


def _lora_dist_moe_weight_references(
    module: DistMoeRoutedExperts,
) -> tuple[torch.Tensor, torch.Tensor]:
    w13_E2FD = module.w13.weight
    w13_lora_a_ELD = module.w13_lora_a.weight
    w13_lora_b_E2FL = module.w13_lora_b.weight
    w2_EDF = module.w2.weight
    w2_lora_a_ELF = module.w2_lora_a.weight
    w2_lora_b_EDL = module.w2_lora_b.weight
    w13_reference_EFD = w13_E2FD.flatten(1, 2) + 2.0 * torch.bmm(
        w13_lora_b_E2FL.flatten(1, 2),
        w13_lora_a_ELD,
    )
    w2_reference_EDF = w2_EDF + 2.0 * torch.bmm(
        w2_lora_b_EDL,
        w2_lora_a_ELF,
    )
    return w13_reference_EFD, w2_reference_EDF


def _runtime() -> DistMoeRuntime:
    runtime = object.__new__(DistMoeRuntime)
    runtime.config = DistMoeRuntime.Config()
    runtime.context = None
    runtime.context_device = torch.device("cuda")
    runtime.ep_pg = cast(Any, object())
    runtime._modules = ()
    runtime._context_config = cast(Any, object())
    runtime.pp_activation_slot_by_stage_and_microbatch = {}
    return runtime


class _NativePostprocess(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-8
        gain_center: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(config.dim))
        self.eps = config.eps
        self.gain_center = config.gain_center

    def forward(self, value_D: torch.Tensor) -> torch.Tensor:
        return value_D

    def to_dist_moe_postprocess(self) -> dist_moe.RMSNormPostprocess:
        """Bind this module's current parameter to the annex descriptor."""
        return dist_moe.RMSNormPostprocess(
            eps=self.eps,
            norm_output_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            weight=self.weight,
            gain_center=self.gain_center,
        )


class _SpecializedGroupedLinear(GroupedLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass


def test_runtime_initializes_and_closes_context_once() -> None:
    """Runtime context construction and teardown are idempotent."""
    runtime = _runtime()
    context = Mock()
    with patch(
        "torchtitan.models.common.dist_moe.dist_moe.create_context",
        return_value=context,
    ) as create:
        runtime.initialize()
        runtime.initialize()

    create.assert_called_once_with(
        group=runtime.ep_pg,
        config=runtime._context_config,
        device=runtime.context_device,
    )
    runtime.close()
    runtime.close()
    context.close.assert_called_once_with()


def test_runtime_selects_pp_activation_slot_from_forward_context() -> None:
    """Pipeline metadata selects the precomputed annex activation slot."""
    runtime = _runtime()
    runtime.pp_activation_slot_by_stage_and_microbatch[(3, 7)] = (2, 5)
    runtime.context = Mock()

    assert runtime.forward_context_key(
        PipelineStageInfo(stage_index=3, microbatch_index=7)
    ) == (2, 5)
    with runtime.forward_context(PipelineStageInfo(stage_index=3, microbatch_index=7)):
        runtime.context.select_activation_slot.assert_called_once_with(2, 5)


def test_engine_owns_runtime_forward_context_and_cleanup() -> None:
    """The generic engine lifecycle registers and removes eager PP contexts."""
    runtime = Mock()
    runtime.forward_context.return_value = nullcontext()
    runtime.forward_context_key.return_value = (2, 5)
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    stage_handle = Mock()
    stage = Mock()
    stage.register_forward_context.return_value = stage_handle

    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(runtimes=[runtime_config])
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    runtime_schedule = SimpleNamespace(_stages=[stage])
    liveness_schedule = SimpleNamespace()
    engine.pp_schedule = SimpleNamespace(
        pipeline_schedule=runtime_schedule,
        pipeline_liveness_schedule=liveness_schedule,
    )
    engine.device = torch.device("cuda")
    engine.runtimes = []
    engine._runtime_stack = ExitStack()

    engine._prepare_training_runtimes()
    info = PipelineStageInfo(stage_index=1, microbatch_index=2)
    forward_context = stage.register_forward_context.call_args.args[0]
    with forward_context(info):
        pass
    engine._close_training_runtimes()

    runtime_config.build.assert_called_once()
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is liveness_schedule
    assert (
        runtime_config.build.call_args.kwargs["parallelism_context"]
        is engine.parallelism_context
    )
    runtime.forward_context.assert_called_once_with(info)
    assert forward_context.graph_cache_key(info) == ((2, 5),)
    stage_handle.remove.assert_called_once_with()
    runtime.close.assert_called_once_with()


def test_transform_rejects_specialized_routed_experts() -> None:
    """Dist-MoE refuses an already-specialized routed-expert implementation."""

    @dataclass(kw_only=True, slots=True)
    class SpecializedConfig(RoutedExperts.Config):
        extra_policy: bool = True

    stock = _stock_config()
    specialized = SpecializedConfig(
        w13=stock.w13,
        w2=stock.w2,
        activation_fn=stock.activation_fn,
        token_dispatcher=stock.token_dispatcher,
    )

    with pytest.raises(TypeError, match="unmodified RoutedExperts.Config"):
        DistMoeTransform().transform(specialized)


def test_transform_rejects_postprocess_without_native_translation() -> None:
    """Unsupported postprocessors fail before model construction."""
    stock = _stock_config()
    stock.output_postprocess = RMSNorm.Config(normalized_shape=32)

    with pytest.raises(TypeError, match="cannot execute inside Dist-MoE"):
        DistMoeTransform().transform(stock)


@pytest.mark.parametrize("projection_name", ["w13", "w2"])
def test_dist_moe_rejects_specialized_projection_config(
    projection_name: str,
) -> None:
    """Owner checks reject configs for specialized projection implementations."""
    stock = _stock_config()
    projection = getattr(stock, projection_name)
    specialized = _SpecializedGroupedLinear.Config(
        param_init=projection.param_init,
        sharding_config=projection.sharding_config,
        group_size=projection.group_size,
        in_features=projection.in_features,
        out_features=projection.out_features,
        num_linears=projection.num_linears,
    )
    setattr(stock, projection_name, specialized)

    with pytest.raises(TypeError, match="stock GroupedLinear W13/W2 projections"):
        DistMoeTransform().transform(stock)


@pytest.mark.parametrize("projection_name", ["w13", "w2"])
def test_dist_moe_rejects_grouped_lora_projection_config(
    projection_name: str,
) -> None:
    """Dist-MoE rejects grouped-LoRA configs owned by adapter implementations."""
    stock = _stock_config()
    projection = getattr(stock, projection_name)
    lora_projection = LoRATransform(
        handlers=(GroupedLinearLoRAHandler(),),
    ).transform(projection)
    assert lora_projection._owner is not GroupedLinear
    setattr(stock, projection_name, lora_projection)

    with pytest.raises(TypeError, match="stock GroupedLinear W13/W2 projections"):
        DistMoeTransform().transform(stock)


def test_dist_moe_rejects_non_swiglu_config() -> None:
    """Dist-MoE continues to require the exact stock SwiGLU config."""
    stock = _stock_config()
    stock.activation_fn = SiTUGLU.Config()

    with pytest.raises(TypeError, match="and SwiGLU"):
        DistMoeTransform().transform(stock)


def test_dist_moe_accepts_frozen_stock_projection_configs() -> None:
    """LoRA freezing preserves the stock Dist-MoE projection contract."""
    transformed = DistMoeTransform().transform(_stock_config())
    transformed = LoRATransform(
        handlers=(LinearLoRAHandler(),),
        target_modules=[],
    ).transform(transformed)

    assert type(transformed.w13) is not GroupedLinear.Config
    assert type(transformed.w2) is not GroupedLinear.Config
    assert transformed.w13._owner is GroupedLinear
    assert transformed.w2._owner is GroupedLinear
    assert type(transformed.activation_fn) is SwiGLU.Config

    module = transformed.build()
    module.init_states()
    assert not module.w13.weight.requires_grad
    assert not module.w2.weight.requires_grad


def test_bf16_transform_preserves_parameter_layout() -> None:
    """BF16 replacement preserves standard W13/W2 checkpoint keys and values."""
    stock = _stock_config().build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    transformed = DistMoeTransform().transform(_stock_config())
    assert isinstance(transformed, DistMoeRoutedExperts.Config)
    module = transformed.build()
    module.load_state_dict(stock.state_dict())

    assert list(dict(module.named_parameters())) == ["w13.weight", "w2.weight"]
    assert not hasattr(module, "token_dispatcher")
    assert not hasattr(module, "activation_fn")
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, stock.state_dict()[key], rtol=0, atol=0)


def test_lora_dist_moe_factory_is_cached_for_exact_parent() -> None:
    """The public factory reuses one adapter class for BF16 Dist-MoE."""
    factory = _get_lora_dist_moe_routed_experts()

    first = factory(DistMoeRoutedExperts)
    second = factory(DistMoeRoutedExperts)

    assert first is second
    assert issubclass(first, DistMoeRoutedExperts)


def test_lora_dist_moe_factory_rejects_mxfp8_parent() -> None:
    """Weight-materializing LoRA does not wrap prepared MXFP8 operands."""
    factory = _get_lora_dist_moe_routed_experts()

    with pytest.raises(ValueError, match="MXFP8DistMoeRoutedExperts"):
        factory(MXFP8DistMoeRoutedExperts)


def test_lora_dist_moe_direct_config_builds_expected_adapter_state() -> None:
    """A direct generated config freezes the base and initializes four adapters."""
    postprocess = _NativePostprocess.Config(
        dim=32,
        param_init={"weight": torch.nn.init.ones_},
    )
    module = _build_lora_dist_moe(output_postprocess=postprocess)

    assert isinstance(module, DistMoeRoutedExperts)
    parameters = dict(module.named_parameters())
    assert set(parameters) == {
        "w13.weight",
        "w2.weight",
        "output_postprocess.weight",
        "w13_lora_a.weight",
        "w13_lora_b.weight",
        "w2_lora_a.weight",
        "w2_lora_b.weight",
    }
    assert {
        name for name, parameter in parameters.items() if parameter.requires_grad
    } == {
        "w13_lora_a.weight",
        "w13_lora_b.weight",
        "w2_lora_a.weight",
        "w2_lora_b.weight",
    }
    assert module.w13_lora_a.weight.shape == (4, 8, 32)
    assert module.w13_lora_b.weight.shape == (4, 2, 64, 8)
    assert module.w2_lora_a.weight.shape == (4, 8, 64)
    assert module.w2_lora_b.weight.shape == (4, 32, 8)
    assert torch.count_nonzero(module.w13_lora_b.weight) == 0
    assert torch.count_nonzero(module.w2_lora_b.weight) == 0


def test_lora_dist_moe_weight_operands_match_independent_references() -> None:
    """W13 and W2 operands include their expert-specific low-rank updates."""
    module = _build_lora_dist_moe()
    _fill_lora_dist_moe_operands(module)

    actual_w13_EFD, actual_w2_EDF = module._weight_operands()
    expected_w13_EFD, expected_w2_EDF = _lora_dist_moe_weight_references(module)

    assert actual_w13_EFD.shape == (4, 128, 32)
    assert actual_w2_EDF.shape == (4, 32, 64)
    torch.testing.assert_close(actual_w13_EFD, expected_w13_EFD)
    torch.testing.assert_close(actual_w2_EDF, expected_w2_EDF)


def test_lora_dist_moe_zero_b_operands_are_exact_base_weights() -> None:
    """Zero-initialized B adapters leave both base operands bitwise unchanged."""
    module = _build_lora_dist_moe()
    with torch.no_grad():
        module.w13.weight.copy_(torch.randn_like(module.w13.weight))
        module.w2.weight.copy_(torch.randn_like(module.w2.weight))

    actual_w13_EFD, actual_w2_EDF = module._weight_operands()

    torch.testing.assert_close(
        actual_w13_EFD,
        module.w13.weight.flatten(1, 2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_w2_EDF,
        module.w2.weight,
        rtol=0,
        atol=0,
    )


def test_lora_dist_moe_weight_operands_backpropagate_only_to_adapters() -> None:
    """Local materialization gives every adapter a finite nonzero gradient."""
    module = _build_lora_dist_moe()
    with torch.no_grad():
        module.w13_lora_a.weight.fill_(0.25)
        module.w13_lora_b.weight.fill_(0.5)
        module.w2_lora_a.weight.fill_(0.75)
        module.w2_lora_b.weight.fill_(1.0)

    w13_EFD, w2_EDF = module._weight_operands()
    w13_gradient_EFD = torch.linspace(
        0.1,
        1.0,
        w13_EFD.numel(),
        dtype=w13_EFD.dtype,
    ).reshape_as(w13_EFD)
    w2_gradient_EDF = torch.linspace(
        1.1,
        2.0,
        w2_EDF.numel(),
        dtype=w2_EDF.dtype,
    ).reshape_as(w2_EDF)
    torch.autograd.backward(
        (w13_EFD, w2_EDF),
        (w13_gradient_EFD, w2_gradient_EDF),
    )

    assert module.w13.weight.grad is None
    assert module.w2.weight.grad is None
    for name in (
        "w13_lora_a.weight",
        "w13_lora_b.weight",
        "w2_lora_a.weight",
        "w2_lora_b.weight",
    ):
        gradient = module.get_parameter(name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == gradient.numel()


def test_lora_dist_moe_forward_passes_effective_weights_without_inplace_wgrad() -> None:
    """Forward gives the annex merged operands and requests functional WGRAD."""
    module = _build_lora_dist_moe()
    _fill_lora_dist_moe_operands(module)
    expected_w13_EFD, expected_w2_EDF = _lora_dist_moe_weight_references(module)
    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())

    with (
        patch(
            "torchtitan.models.common.dist_moe.dist_moe.routed_experts",
            return_value=torch.empty(2, 32),
        ) as execute,
        patch(
            "torchtitan.models.common.dist_moe.remat.region",
            side_effect=lambda fn, *_args, **_kwargs: fn,
        ),
        patch("torchtitan.models.common.dist_moe.remat.recompute_needs_tensor"),
    ):
        module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
        )

    passed_w13_EFD = execute.call_args.args[3]
    passed_w2_EDF = execute.call_args.args[4]
    options = execute.call_args.kwargs["options"]
    torch.testing.assert_close(passed_w13_EFD, expected_w13_EFD)
    torch.testing.assert_close(passed_w2_EDF, expected_w2_EDF)
    assert passed_w13_EFD.data_ptr() != module.w13.weight.data_ptr()
    assert passed_w2_EDF.data_ptr() != module.w2.weight.data_ptr()
    assert options.inplace_wgrad_accum is False


def test_dist_moe_lora_handler_converts_exact_bf16_config() -> None:
    """The parent handler selects the BF16 effective-weight LoRA class."""
    handler = _get_dist_moe_lora_handler()()
    base_config = cast(
        DistMoeRoutedExperts.Config,
        DistMoeTransform().transform(_stock_config()),
    )

    transformed = handler.make_config(
        base_config,
        rank=8,
        alpha=16.0,
    )

    lora_cls = _get_lora_dist_moe_routed_experts()(DistMoeRoutedExperts)
    assert handler.config_type is DistMoeRoutedExperts.Config
    assert type(transformed) is lora_cls.Config
    assert transformed.rank == 8
    assert transformed.alpha == 16.0
    assert transformed.w13 is base_config.w13
    assert transformed.w2 is base_config.w2


def test_dist_moe_lora_handler_rejects_inplace_wgrad() -> None:
    """Transient effective weights require functional Dist-MoE WGRAD."""
    base_config = cast(
        DistMoeRoutedExperts.Config,
        DistMoeTransform(inplace_wgrad_accum=True).transform(_stock_config()),
    )

    with pytest.raises(ValueError, match="inplace_wgrad_accum"):
        _get_dist_moe_lora_handler()().make_config(
            base_config,
            rank=8,
            alpha=16.0,
        )


def test_dist_moe_lora_handler_rejects_mxfp8_before_build() -> None:
    """Prepared MXFP8 operands are rejected at config transformation time."""
    base_config = MXFP8DistMoeTransform().transform(_stock_config())

    with pytest.raises(ValueError, match="MXFP8DistMoeRoutedExperts"):
        _get_dist_moe_lora_handler()().make_config(
            base_config,
            rank=8,
            alpha=16.0,
        )


def test_dist_moe_lora_parent_target_adapts_both_projections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One routed-experts target owns both effective-weight adapters."""
    model_config = deepseek_v3_debugmodel().model
    caplog.set_level("WARNING", logger="torchtitan.config.transform.lora")

    transformed = transform_api.transform_model_config_(
        model_config,
        [
            DistMoeTransform(),
            LoRATransform(
                handlers=(_get_dist_moe_lora_handler()(),),
                rank=8,
                alpha=16.0,
                target_modules=["routed_experts"],
            ),
        ],
    )

    lora_cls = _get_lora_dist_moe_routed_experts()(DistMoeRoutedExperts)
    routed_configs = list(transformed.traverse(DistMoeRoutedExperts.Config))
    assert len(routed_configs) == 5
    for _fqn, config, _parent, _attr in routed_configs:
        assert config._owner is lora_cls
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.w13._owner is GroupedLinear
        assert config.w2._owner is GroupedLinear
        assert not hasattr(config.w13, "rank")
        assert not hasattr(config.w2, "rank")
    assert not any("did not match" in record.message for record in caplog.records)


def test_dist_moe_lora_all_targets_adapt_every_routed_experts_parent() -> None:
    """The Dist-MoE handler's all-target mode converts every parent node."""
    transformed = transform_api.transform_model_config_(
        deepseek_v3_debugmodel().model,
        [
            DistMoeTransform(),
            LoRATransform(
                handlers=(_get_dist_moe_lora_handler()(),),
                rank=8,
                alpha=16.0,
            ),
        ],
    )

    lora_cls = _get_lora_dist_moe_routed_experts()(DistMoeRoutedExperts)
    routed_configs = list(transformed.traverse(DistMoeRoutedExperts.Config))
    assert len(routed_configs) == 5
    for _fqn, config, _parent, _attr in routed_configs:
        assert config._owner is lora_cls
        assert type(config.w13) is not GroupedLinear.Config
        assert type(config.w2) is not GroupedLinear.Config
        assert config.w13._owner is GroupedLinear
        assert config.w2._owner is GroupedLinear


def test_dist_moe_lora_rejects_grouped_lora_subtree_conflict() -> None:
    """Parent and child handlers cannot both claim the expert projections."""
    with pytest.raises(
        ValueError,
        match="(?i)(subtree.*grouped|grouped.*subtree)",
    ):
        transform_api.transform_model_config_(
            deepseek_v3_debugmodel().model,
            [
                DistMoeTransform(),
                LoRATransform(
                    handlers=(
                        _get_dist_moe_lora_handler()(),
                        GroupedLinearLoRAHandler(),
                    ),
                    rank=8,
                    alpha=16.0,
                ),
            ],
        )


@pytest.mark.parametrize(
    ("expert_transform", "expected_owner"),
    [
        (DistMoeTransform(), DistMoeRoutedExperts),
        (MXFP8DistMoeTransform(), MXFP8DistMoeRoutedExperts),
    ],
)
def test_dist_moe_lora_dense_target_remains_legal_with_non_target_experts(
    expert_transform,
    expected_owner,
) -> None:
    """Dense-only LoRA leaves either Dist-MoE backend unadapted and frozen."""
    transformed = transform_api.transform_model_config_(
        deepseek_v3_debugmodel().model,
        [
            expert_transform,
            LoRATransform(
                handlers=(LinearLoRAHandler(),),
                rank=8,
                alpha=16.0,
                target_modules=["wo"],
            ),
        ],
    )

    routed_configs = list(transformed.traverse(DistMoeRoutedExperts.Config))
    assert len(routed_configs) == 5
    assert all(config._owner is expected_owner for _, config, _, _ in routed_configs)
    assert all(not hasattr(config, "rank") for _, config, _, _ in routed_configs)
    dense_targets = [
        config
        for fqn, config, _parent, _attr in transformed.traverse(Linear.Config)
        if fqn.endswith(".wo")
    ]
    assert len(dense_targets) == 6
    assert all(hasattr(config, "rank") for config in dense_targets)


def test_dist_moe_stock_weight_operands_preserve_parameter_identity() -> None:
    """The stock BF16 path keeps its zero-copy W13 view and exact W2 object."""
    config = cast(
        DistMoeRoutedExperts.Config, DistMoeTransform().transform(_stock_config())
    )
    module = cast(DistMoeRoutedExperts, config.build())
    module.init_states()

    w13_EFD, w2_EDF = module._weight_operands()

    assert w13_EFD.data_ptr() == module.w13.weight.data_ptr()
    assert w2_EDF is module.w2.weight

    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())
    with (
        patch(
            "torchtitan.models.common.dist_moe.dist_moe.routed_experts",
            return_value=torch.empty(2, 32),
        ) as execute,
        patch(
            "torchtitan.models.common.dist_moe.remat.region",
            side_effect=lambda fn, *_args, **_kwargs: fn,
        ),
        patch("torchtitan.models.common.dist_moe.remat.recompute_needs_tensor"),
    ):
        module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
        )

    passed_w13_EFD = execute.call_args.args[3]
    passed_w2_EDF = execute.call_args.args[4]
    assert passed_w13_EFD.data_ptr() == module.w13.weight.data_ptr()
    assert passed_w2_EDF is module.w2.weight


def test_runtime_passes_per_slot_capacity_to_annex() -> None:
    """Runtime preserves the annex's per-slot activation-capacity contract."""
    vmm = dist_moe.VmmConfig(total_scratch_capacity_factor=4.0, prefetch=False)
    transformed = DistMoeTransform(
        runtime=DistMoeRuntime.Config(
            device_scratch_capacity_factor=2.0,
            activation_slot_bytes=2048,
            vmm=vmm,
        ),
        bf16_grouped_gemm_preset="1cta1mma_bm64_bn128",
    ).transform(_stock_config(dim=64))
    module = cast(DistMoeRoutedExperts, transformed.build())
    runtime = _runtime()
    runtime.config = DistMoeRuntime.Config(
        device_scratch_capacity_factor=2.0,
        activation_slot_bytes=2048,
        vmm=vmm,
    )

    context_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )

    assert context_config.max_local_input_tokens == 128
    assert context_config.max_moe_layers_per_activation_slot == 3
    assert context_config.device_scratch_capacity_factor == 2.0
    assert context_config.activation_slot_bytes == 2048
    assert context_config.activation_slot_capacity_factor is None
    assert context_config.num_activation_slots == 2
    assert context_config.vmm is vmm
    assert context_config.bf16_grouped_gemm_preset == "1cta1mma_bm64_bn128"

    runtime.config = DistMoeRuntime.Config(
        device_scratch_capacity_factor=2.0,
        activation_slot_capacity_factor=1.5,
        vmm=vmm,
    )
    factor_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )
    assert factor_config.activation_slot_bytes is None
    assert factor_config.activation_slot_capacity_factor == 1.5
    assert factor_config.num_activation_slots == 2


def test_mxfp8_transform_is_independent_and_uses_prepared_weights() -> None:
    """MXFP8 transforms stock experts directly and installs prepared weights."""
    transformed = MXFP8DistMoeTransform().transform(_stock_config())
    assert isinstance(transformed, MXFP8DistMoeRoutedExperts.Config)

    module = transformed.build()
    assert isinstance(module, MXFP8DistMoeRoutedExperts)
    assert isinstance(module.w13.weight, _ShardedFSDPTensor)
    assert isinstance(module.w2.weight, _ShardedFSDPTensor)
    assert list(module.state_dict()) == ["w13.weight", "w2.weight"]


def test_dist_moe_transforms_conflict() -> None:
    """A routed-expert module cannot select BF16 and MXFP8 Dist-MoE together."""
    config = deepseek_v3_debugmodel()
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_transforms(config, [DistMoeTransform(), MXFP8DistMoeTransform()])


def test_transform_registers_one_runtime() -> None:
    """Applying Dist-MoE records one generic rank-wide runtime configuration."""
    config = deepseek_v3_debugmodel()
    transformed = apply_transforms(config, [DistMoeTransform()])
    assert len(transformed.runtimes) == 1
    assert isinstance(transformed.runtimes[0], DistMoeRuntime.Config)


def test_forward_passes_native_postprocess_and_wgrad_policy() -> None:
    """Forward passes module-owned postprocessing and annex-owned WGRAD policy."""
    stock = _stock_config()
    stock.output_postprocess = _NativePostprocess.Config(dim=32)
    transformed = DistMoeTransform(inplace_wgrad_accum=True).transform(stock)
    module = cast(DistMoeRoutedExperts, transformed.build())
    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())

    with (
        patch(
            "torchtitan.models.common.dist_moe.dist_moe.routed_experts",
            return_value=torch.empty(2, 32),
        ) as execute,
        patch(
            "torchtitan.models.common.dist_moe.remat.region",
            side_effect=lambda fn, *_args, **_kwargs: fn,
        ) as remat_region,
        patch(
            "torchtitan.models.common.dist_moe.remat.recompute_needs_tensor"
        ) as recompute_needs_tensor,
    ):
        out_TD = module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
        )

    options = execute.call_args.kwargs["options"]
    descriptor = options.experts_output_postprocess
    assert isinstance(descriptor, dist_moe.RMSNormPostprocess)
    assert descriptor.weight is module.output_postprocess.weight
    assert options.inplace_wgrad_accum
    assert options.wgrad_parameter_owners is None
    assert "output_postprocess.weight" in module.state_dict()
    assert remat_region.call_args.kwargs == {"recompute": False}
    recompute_needs_tensor.assert_called_once_with(out_TD)


@pytest.mark.parametrize(
    "kwargs,error_type,message",
    [
        ({"device_scratch_capacity_factor": 0}, ValueError, "must be positive"),
        ({"activation_slot_bytes": -1}, ValueError, "cannot be negative"),
        ({"activation_slot_bytes": True}, TypeError, "must be an integer"),
        ({"activation_slot_capacity_factor": -1}, ValueError, "nonnegative"),
        ({"activation_slot_capacity_factor": float("nan")}, ValueError, "finite"),
        (
            {
                "activation_slot_bytes": 1,
                "activation_slot_capacity_factor": 1.0,
            },
            ValueError,
            "mutually exclusive",
        ),
        (
            {"pp_activation_slot_policy": "invalid"},
            ValueError,
            "activation-slot policy",
        ),
        ({"wgrad_dtype": "float16"}, ValueError, "WGRAD dtype"),
    ],
)
def test_runtime_config_rejects_invalid_values(kwargs, error_type, message) -> None:
    """Rank-wide runtime configuration rejects invalid memory policies."""
    with pytest.raises(error_type, match=message):
        DistMoeRuntime.Config(**kwargs)


def test_runtime_config_requires_bfloat16_unsharded_parameters() -> None:
    """Dist-MoE rejects FSDP mixed-precision parameter dtypes it cannot consume."""
    with pytest.raises(ValueError, match="mixed_precision_param='bfloat16'"):
        DistMoeRuntime.Config().validate(
            SimpleNamespace(training=SimpleNamespace(mixed_precision_param="float32"))
        )
