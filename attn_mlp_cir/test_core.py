from types import SimpleNamespace

import torch

from attn_mlp_cir.attention_ov import AttentionOVSkeleton
from attn_mlp_cir.audit import inspect_norm
from attn_mlp_cir.cache import AttentionOVCache
from attn_mlp_cir.mlp import MLPSkeleton
from attn_mlp_cir.projections import AttentionProjections, resolve_attention_projections
from attn_mlp_cir.types import PathSelection


def _projections(d_model=7, qh=4, kvh=2, hd=3, bias=False):
    torch.manual_seed(3)
    return AttentionProjections(
        value=torch.randn(d_model, kvh, hd),
        output=torch.randn(qh, hd, d_model),
        value_bias=torch.randn(kvh, hd) if bias else None,
        output_bias=torch.randn(d_model) if bias else None,
        query_to_kv=torch.arange(qh) // (qh // kvh),
    )


def test_full_source_resolved_ov_reconstruction_gqa():
    p = _projections(bias=True)
    skeleton = AttentionOVSkeleton.from_projections(p)
    assert skeleton.rank == p.head_dim
    x = torch.randn(2, 5, 7)
    logits = torch.randn(2, 4, 3, 5)
    probs = logits.softmax(dim=-1)
    actual_values = torch.einsum("bsd,dkc->bskc", x, p.value) + p.value_bias
    q_values = actual_values.index_select(-2, p.query_to_kv)
    attended = torch.einsum("bhqs,bshd->bqhd", probs, q_values)
    expected = torch.einsum("bqhd,hdo->bqo", attended, p.output) + p.output_bias
    torch.testing.assert_close(
        skeleton.reconstruct(x, probs), expected, rtol=3e-5, atol=3e-5
    )


def test_target_scores_sum_to_target_projection_without_bias():
    p = _projections()
    skeleton = AttentionOVSkeleton.from_projections(p)
    x = torch.randn(2, 5, 7)
    probs = torch.randn(2, 4, 3, 5).softmax(-1)
    target = torch.randn(7)
    scores = skeleton.path_scores(x, probs, target)
    expected = skeleton.reconstruct(x, probs, include_constants=False) @ target
    torch.testing.assert_close(scores.sum(dim=(-3, -2, -1)), expected)


def test_batched_attention_targets_match_independent_scoring():
    skeleton = AttentionOVSkeleton.from_projections(_projections())
    x = torch.randn(3, 5, 7)
    probs = torch.randn(3, 4, 2, 5).softmax(-1)
    targets = torch.randn(3, 7)
    batched = skeleton.aggregated_scores(x, probs, targets)
    independent = torch.stack([
        skeleton.aggregated_scores(x[i], probs[i], targets[i]) for i in range(3)
    ])
    torch.testing.assert_close(batched, independent)


def test_coarse_head_mode_scores_reconstruct_after_qk_aggregation():
    skeleton = AttentionOVSkeleton.from_projections(_projections())
    x = torch.randn(2, 5, 7)
    probs = torch.randn(2, 4, 3, 5).softmax(-1)
    target = torch.randn(7)
    scores = skeleton.aggregated_scores(x, probs, target)
    expected = skeleton.reconstruct_aggregated(
        x, probs, include_constants=False
    ) @ target
    torch.testing.assert_close(scores.sum(dim=(-2, -1)), expected, rtol=3e-5, atol=3e-5)


def test_global_and_per_head_budgets():
    skeleton = AttentionOVSkeleton.from_projections(_projections())
    scores = torch.randn(2, 3, 5, 4, skeleton.rank)
    global_trace = skeleton.select(scores, 6, PathSelection.GLOBAL)
    assert torch.all(global_trace.mask.sum(dim=(-3, -2, -1)) == 6)
    head_trace = skeleton.select(scores, 6, PathSelection.PER_HEAD)
    assert torch.all(head_trace.mask.sum(dim=(-3, -1)) == 6)


def test_attention_topk_uses_signed_descending_order():
    skeleton = AttentionOVSkeleton.from_projections(_projections())
    scores = torch.tensor([[[[-100.0, 3.0, 2.0], [-4.0, 1.0, -0.5]]]])
    trace = skeleton.select_aggregated(scores, 4, PathSelection.GLOBAL)
    selected = scores[trace.mask.bool()]
    torch.testing.assert_close(selected.sort().values, torch.tensor([-0.5, 1.0, 2.0, 3.0]))


def test_mlp_topk_uses_signed_descending_order():
    from attn_mlp_cir.forward import _mlp_mask

    scores = torch.tensor([[[-100.0, 3.0, 2.0, -4.0, 1.0]]])
    mask = _mlp_mask(scores, 4, "topk", seed=0)
    selected = scores[mask.bool()]
    torch.testing.assert_close(selected.sort().values, torch.tensor([-4.0, 1.0, 2.0, 3.0]))


def test_mlp_full_reconstruction():
    layer = torch.nn.Linear(9, 6)
    skeleton = MLPSkeleton.from_output_projection(layer)
    h = torch.randn(2, 4, 9)
    torch.testing.assert_close(skeleton.reconstruct(h), layer(h), rtol=2e-5, atol=2e-5)


def test_batched_mlp_targets_match_independent_scoring():
    skeleton = MLPSkeleton.from_output_projection(torch.nn.Linear(9, 6))
    h = torch.randn(3, 1, 9)
    targets = torch.randn(3, 6)
    batched = skeleton.path_scores(h, targets)
    independent = torch.stack([
        skeleton.path_scores(h[i], targets[i]) for i in range(3)
    ])
    torch.testing.assert_close(batched, independent)


def test_mlp_effective_target_applies_gemma_post_ffn_gamma_only():
    from attn_mlp_cir.forward import _mlp_effective_target

    target = torch.tensor([[2.0, -3.0]])
    ordinary_block = SimpleNamespace()
    assert _mlp_effective_target(ordinary_block, target) is target

    gemma_block = SimpleNamespace(
        post_feedforward_layernorm=SimpleNamespace(
            weight=torch.tensor([0.25, -0.5])
        )
    )
    torch.testing.assert_close(
        _mlp_effective_target(gemma_block, target),
        target * torch.tensor([1.25, 0.5]),
    )


class _RMS(torch.nn.Module):
    def __init__(self, width, offset=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.2, 0.8, width))
        self.variance_epsilon = 1e-6
        self.offset = offset

    def forward(self, x):
        normed = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)
        scale = 1 + self.weight if self.offset else self.weight
        return normed.to(x.dtype) * scale


def test_rmsnorm_semantics_are_measured_not_guessed_from_name():
    assert inspect_norm(_RMS(8), "x").affine == "weight"
    assert inspect_norm(_RMS(8, offset=True), "x").affine == "one_plus_weight"


def test_attention_cache_round_trip(tmp_path):
    import json
    projections = _projections(bias=True)
    skeleton = AttentionOVSkeleton.from_projections(projections)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "format_version": 1, "model_name": "dummy", "n_layers": 1,
    }))
    torch.save({
        "model_name": "dummy", "layer_idx": 0,
        "U": skeleton.U, "S": skeleton.S, "V": skeleton.V,
        "constant": skeleton.constant, "output_bias": skeleton.output_bias,
    }, tmp_path / "layer_000.pt")
    loaded = AttentionOVCache(str(tmp_path), "dummy").skeleton(0)
    x = torch.randn(2, 5, 7)
    probs = torch.randn(2, 4, 3, 5).softmax(-1)
    torch.testing.assert_close(
        loaded.reconstruct_aggregated(x, probs),
        skeleton.reconstruct_aggregated(x, probs),
    )


def test_gpt_neo_attention_wrapper_resolves_v_and_output_projections():
    from transformers import GPTNeoConfig, GPTNeoForCausalLM

    config = GPTNeoConfig(
        vocab_size=31,
        max_position_embeddings=16,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        attention_types=[[['global'], 1]],
    )
    model = GPTNeoForCausalLM(config)
    inner = model.transformer.h[0].attn.attention
    projections = resolve_attention_projections(model, 0)

    assert projections.value.shape == (16, 4, 4)
    assert projections.output.shape == (4, 4, 16)
    torch.testing.assert_close(
        projections.value.reshape(16, 16), inner.v_proj.weight.detach().T
    )
    torch.testing.assert_close(
        projections.output.reshape(16, 16), inner.out_proj.weight.detach().T
    )

    # The legacy attention selector uses a separate W_O resolver and must also
    # descend through the GPTNeoAttention wrapper.
    from circuit_skeleton.attention_skeleton import get_attention_output_projection
    output, linear_layout, heads, head_dim = get_attention_output_projection(model, 0)
    assert output is inner.out_proj
    assert linear_layout is True
    assert (heads, head_dim) == (4, 4)


def test_multilayer_gemma2_eager_full_circuit_matches_native_forward():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from attn_mlp_cir.forward import CircuitForwardConfig, DynamicCircuitForward
    from attn_mlp_cir.types import CircuitMode

    torch.manual_seed(7)
    config = Gemma2Config(
        vocab_size=41,
        max_position_embeddings=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=3,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    model = Gemma2ForCausalLM(config).eval()
    model.config._attn_implementation = "eager"
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    with torch.no_grad():
        native = model(input_ids, use_cache=False).logits

    runner = DynamicCircuitForward(model)
    circuit = runner(
        input_ids,
        model.lm_head.weight[[11, 12]].detach(),
        CircuitForwardConfig(
            start_layer=0,
            end_layer=2,
            mlp_k=10**9,
            attention_k=10**9,
            mode=CircuitMode.ATTENTION_MLP,
            post_norm_mode="native",
        ),
    )
    torch.testing.assert_close(circuit, native, rtol=2e-5, atol=2e-5)
