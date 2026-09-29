# Distributed MoE

TorchTitan can replace standard routed experts with the CuTe DSL
[`dist-moe`](https://github.com/meta-pytorch/dist_moe) backend. The router stays
in TorchTitan and produces top-k expert IDs and scores. Dist-MoE owns token
dispatch, grouped expert computation, activation storage, scratch storage, and
combine as one ordered operation.

TorchTitan supports BF16 and asynchronous MXFP8 Dist-MoE training on NVIDIA
SM100 or newer. The annex also provides NVFP4 inference, which TorchTitan does
not expose yet. Dist-MoE requires
`training.mixed_precision_param="bfloat16"` because FSDP unshards its persistent
parameters in BF16.

## Select A Backend

The BF16 and MXFP8 transforms are independent. Apply exactly one of them to a
completed model configuration:

```python
from torchtitan.config.transform import apply_transforms, DistMoeTransform
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b

config = apply_transforms(
    deepseek_v3_16b(),
    [
        DistMoeTransform(
            runtime=DistMoeRuntime.Config(
                device_scratch_capacity_factor=4.0,
                activation_slot_bytes=None,
                pp_activation_slot_policy="stage_microbatch",
            )
        )
    ],
)
```

For MXFP8 experts, replace `DistMoeTransform` with
`MXFP8DistMoeTransform`. Dense attention, shared-expert, feed-forward, and
language-model-head linears are separate quantization choices:

```python
import dist_moe

from torchtitan.config.transform import apply_transforms, MXFP8DistMoeTransform
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b

config = apply_transforms(
    deepseek_v3_16b(),
    [
        MXFP8DistMoeTransform(
            runtime=DistMoeRuntime.Config(
                device_scratch_capacity_factor=4.0,
            ),
            block_scaled_config=dist_moe.BlockScaledConfig(
                pipeline="staged",
                fast_math=False,
            ),
        )
    ],
)
```

DeepSeek V3 reference recipes are available for the debug, 16B, and 671B
models with `_dist_moe_bf16` and `_dist_moe_mxfp8` suffixes. They select
CUDA-graph-compatible varlen attention. The MXFP8 recipes also apply
TorchTitan's existing MXFP8 linear conversion to dense projections and the
language-model head.

## Configuration Ownership

`DistMoeTransform` and `MXFP8DistMoeTransform` replace each stock
`RoutedExperts.Config` directly. The replacement keeps TorchTitan's structured
`GroupedLinear` parameter layout:

- `w13.weight` has shape `[E, 2, F, D]`.
- `w2.weight` has shape `[E, D, F]`.
- An optional `output_postprocess` remains a normal TorchTitan module.

This preserves parameter, optimizer, FSDP, and checkpoint ownership. The
transformed module does not construct the stock activation or token dispatcher,
because the annex performs those operations internally.

Per-layer settings belong to the transform:

| Setting | Meaning |
| --- | --- |
| `inplace_wgrad_accum` | Let annex WGRAD kernels accumulate into an existing standard `parameter.grad` buffer. The annex derives ownership from each logical weight. Enable it only when the integration keeps that gradient storage stable across serialized backward calls. |
| `bf16_grouped_gemm_preset` | Optional BF16 FPROP/DGRAD schedule override for expert users. `None` uses the annex's shape-aware production defaults. BF16 WGRAD has its own production schedule. |
| `block_scaled_config` | MXFP8-only annex policy. `pipeline="staged"` uses separate expert kernels; `pipeline="mega"` uses the fused chunk-pipelined implementation. `fast_math` selects approximate fused-SwiGLU sigmoid math, and `kernel_config` is an expert-only CuTe tuning override. |

One `DistMoeRuntime.Config` owns rank-wide resources shared by every local
Dist-MoE layer:

| Setting | Meaning |
| --- | --- |
| `device_scratch_capacity_factor` | Routing imbalance that must fit entirely in HBM scratch. `1.0` covers balanced `local_tokens * top_k` routing. |
| `activation_slot_bytes` | Exact saved-forward-state capacity requested for each live activation slot, excluding scratch. Mutually exclusive with `activation_slot_capacity_factor`. |
| `activation_slot_capacity_factor` | Per-slot saved-state capacity relative to balanced routing. `1.0` retains every eligible intermediate when aggregate slot usage is balanced. Mutually exclusive with `activation_slot_bytes`; leaving both unset selects the minimum all-recompute plan. |
| `pp_activation_slot_policy` | `"stage_microbatch"` reuses slots at stage-microbatch lifetime; `"microbatch"` retains one deeper slot across all local stages for a microbatch. The default is `"stage_microbatch"`. |
| `vmm` | Optional annex `VmmConfig` for host-backed overflow scratch. Saved activations always remain in HBM. |
| `num_sms` | Optional SM count for each Dist-MoE CuTe launch. `None` uses the annex default. |
| `wgrad_dtype` | W13/W2 gradient output dtype: `"bfloat16"` or `"float32"`. Tensor-core accumulation remains FP32. |

The transform registers this runtime automatically. Application code should not
construct or initialize `DistMoeRuntime` directly.

## Scratch Capacity

Let `T` be the maximum local input tokens after CP and sequence-parallel
sharding, `K` be top-k, and `P` be the expert-parallel degree. Balanced routing
receives `T * K` rows on each rank. The largest possible receive count is:

```text
T * P * min(K, num_local_experts)
```

The corresponding worst-case imbalance relative to balanced routing is:

```text
P * min(K, num_local_experts) / K
```

`device_scratch_capacity_factor` sets the HBM-resident receive and scratch
bound. It does not truncate or rebalance routing. A factor of `1.0` is suitable
for forced-balanced routing. Real routing needs measured headroom; factor four
is a deliberate recipe choice, not a universal default. Block-scaled kernels
also pad each local expert to their M-tile size, so planned rows can be slightly
larger than the logical bound.

Without VMM, exceeding the device factor is an error. With VMM,
`VmmConfig.total_scratch_capacity_factor` is the larger device-plus-host
correctness bound. Exceeding that bound is still an error.

## Saved Activations And Pipeline Slots

The annex allocates four distinct regions:

| Allocation | Placement | Contents and lifetime |
| --- | --- | --- |
| Symmetric communication buffers | Peer-addressable HBM | Routing metadata, signals, dispatch inputs, combine outputs, and gradients for the context lifetime. |
| Saved-activation region | Rank-local HBM | Mandatory layer inputs and any expert intermediates selected for saving, partitioned into live activation slots. |
| Device scratch | Rank-local HBM | Forward temporaries, recompute temporaries, and activation gradients, reused by one local layer action at a time. |
| VMM overflow scratch | Pinned host memory in the same CUDA virtual range | Optional scratch demand above the device factor and within the total factor. |

Without PP there is one activation slot containing all local MoE layers. With
eager PP, PyTorch analyzes the final schedule before warmup and assigns every
live `(stage, microbatch)` interval to a reusable slot. The default
`"stage_microbatch"` policy sizes each slot for the largest local stage and
reuses it after that stage's backward releases the saved state. The
`"microbatch"` policy keeps one slot across all local stages for a microbatch
and therefore sizes each slot for their combined MoE depth.

Activation capacity is configured independently for each live slot. Set
`activation_slot_bytes` for an exact logical byte budget, or use
`activation_slot_capacity_factor` to scale optional saved state above the
mandatory layer inputs. The controls are mutually exclusive. When both are
`None`, the annex selects its minimum correct plan: every assigned layer input
remains saved and other expert intermediates are dynamically recomputed in
backward. Increasing either policy can retain more intermediates and reduce
recomputation up to the planner's logged maximum useful per-slot budget.

PyTorch calls each eager pipeline stage's registered forward context with its
stage and microbatch indices. `DistMoeRuntime` uses the precomputed assignment
to select the annex activation slot before model execution. This does not add
model kwargs or require a custom pipeline-stage subclass.

The annex represents each physical slot with an immutable device-scalar view.
Non-strict FX tracing and whole-step CUDA-graph capture can therefore bind a
fixed view to each scheduled Dist-MoE call without copying or reading a GPU
scalar on the host. GraphPP specializes a stage graph by the runtime forward
context key. For Dist-MoE, that key is the assigned activation-slot ID and the
number of MoE layers using the slot. Microbatches with the same key share one
graph; different keys receive separate graph variants whose traces bind the
corresponding immutable annex view. The overlapped forward/backward GraphPP
action currently requires a single forward-context variant per stage.

The annex's functional and accumulating backward operations are both visible to
non-strict FX tracing. PP GraphPP currently keeps gradient accumulation outside
the stage graphs, so its integration recipes trace functional Dist-MoE WGRAD
outputs. A future GraphTrainer fusion rule can replace those outputs and their
accumulation sinks with the annex's explicit mutating backward operation. That
is a performance follow-up, not a correctness requirement for activation-slot
specialization.

## VMM Overflow Scratch

VMM keeps the fast scratch prefix in HBM and maps additional pinned host pages
into the same stable CUDA virtual range:

```python
import dist_moe

from torchtitan.config.transform import apply_transforms, DistMoeTransform
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b

config = apply_transforms(
    deepseek_v3_16b(),
    [
        DistMoeTransform(
            runtime=DistMoeRuntime.Config(
                device_scratch_capacity_factor=1.0,
                vmm=dist_moe.VmmConfig(
                    total_scratch_capacity_factor=4.0,
                    prefetch=True,
                ),
            )
        )
    ],
)
```

Scratch above factor one and at most factor four can use host-backed pages. No
saved activation is moved to host memory. `prefetch=True` overlaps physical VMM
allocation with communication-buffer initialization during context creation;
`False` performs the same allocation synchronously afterward. Prefetch does not
change capacity, steady-state placement, kernel behavior, or execution-time
CPU/GPU synchronization. Disable it when simpler serialized initialization is
more important than reducing startup latency.

## FSDP, MXFP8, And Rematerialization

BF16 weights use the ordinary FSDP lifecycle. MXFP8 experts reuse TorchTitan's
prepared-weight tensor lifecycle:

1. FSDP all-gathers the persistent high-precision shard in BF16.
2. The post-all-gather hook prepares grouped 32x32 MXFP8 qdata and FPROP/DGRAD
   scale layouts.
3. FSDP releases the temporary BF16 communication storage.
4. Dist-MoE consumes the prepared operands for that unshard lifetime.
5. FSDP releases prepared storage at the normal reshard boundary.

With `reshard_after_forward=False`, pipeline microbatches reuse one prepared
weight through backward. With `reshard_after_forward=True`, forward and
backward perform separate unshards and preparations. Without FSDP, the module
prepares the live parameter directly for each call.

The complete `dist_moe.routed_experts` call is an ordered operation wrapped in
a non-recomputed remat region. Dist-MoE owns its device-side decision to save or
recompute expert intermediates; an outer activation-checkpoint policy must not
duplicate dispatch, communication, or arena mutation.

## Expert Output Postprocessing

The common routed-expert API can own an `output_postprocess` module that runs
after W2 and before score-weighted top-k combine. TorchTitan keeps that module's
parameter, optimizer, checkpoint, and FSDP ownership. At each forward the
Dist-MoE adapter converts its current parameter to the annex's typed
`RMSNormPostprocess` execution descriptor. Unsupported postprocessors fail
during configuration instead of running after combine or falling back to an
unfused callback.

## Runtime Lifecycle

TorchTitan owns optional rank-level runtimes through a generic lifecycle:

1. Model transforms register required runtime configs.
2. After model parallelization and pipeline construction, the training engine
   builds runtimes from the final model parts, device meshes, and schedule.
3. After parameters and buffers materialize, each runtime initializes external
   resources. Dist-MoE creates one annex context and attaches non-owning
   references to all local expert modules.
4. Eager pipeline stages enter the composed runtime forward context.
5. Failure or normal teardown removes forward registrations and closes runtimes
   in reverse construction order.

The annex context owns symmetric buffers, activation storage, scratch storage,
VMM allocation, and VMM prefetch. TorchTitan never accesses their private
representations. See the annex
[memory planner](https://github.com/meta-pytorch/dist_moe/blob/main/docs/memory_planner.md),
[pipeline slots](https://github.com/meta-pytorch/dist_moe/blob/main/docs/pipeline_activation_slots.md),
and [VMM](https://github.com/meta-pytorch/dist_moe/blob/main/docs/vmm.md)
documentation for byte-level layouts and planner behavior.
