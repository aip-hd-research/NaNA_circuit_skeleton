"""
channel_first.py

Channel-first static skeleton: selection + forward passes.

Selection strategy
------------------
1. effector[i] = softmax(V @ W_emb.T)[i, target_id]   — target probability per subspace
2. Sort subspaces by effector[i] descending.
3. For each of the top k_channel subspaces, take top k_neuron neurons by |delta[j,i]|.

Forward pass gating
-------------------
Sign-gate: keep path (j,i) if  h[j] * static_score[j,i] > 0
where  static_score[j,i] = delta[j,i] * effector[i].
Since effector is always positive (softmax output), gate = sign(h[j]) == sign(delta[j,i]).
"""

import json
import os
import torch
from model_config import apply_template, needs_true_add_special_tokens
from ske_pre_valid.within_ske_selector import reconstruct_mlp_via_skeleton
from ske_pre_valid.static_skeleton import target_rank_prob, top_tokens, select_random_paths


# ── selection ─────────────────────────────────────────────────────────────────

def select_channel_first(delta, V, W_emb, target_id, k_channel, k_neuron):
    """
    Channel-first path selection.

    Returns list of path dicts (compatible with reconstruct_mlp_via_skeleton):
        neuron_idx, subspace_idx, delta_val, static_score, abs_score,
        effector, channel_rank, activation, rank
    """
    delta  = delta.cpu()
    V_cpu  = V.cpu()
    W_cpu  = W_emb.cpu()

    logits   = V_cpu @ W_cpu.T                              # [n_sp, vocab]
    effector = torch.softmax(logits, dim=-1)[:, target_id]  # [n_sp], in (0, 1)

    k_ch       = min(k_channel, delta.shape[1])
    top_ch_idx = torch.topk(effector, k_ch).indices         # descending by effector

    paths = []
    for ch_rank, i in enumerate(top_ch_idx.tolist(), 1):
        eff_i = effector[i].item()
        k_n   = min(k_neuron, delta.shape[0])
        top_j = torch.topk(delta[:, i].abs(), k_n).indices.tolist()
        for j in top_j:
            dv = delta[j, i].item()
            paths.append({
                "neuron_idx":   j,
                "subspace_idx": i,
                "delta_val":    dv,
                "static_score": dv * eff_i,
                "abs_score":    abs(dv * eff_i),
                "effector":     eff_i,
                "channel_rank": ch_rank,
                "activation":   None,
                "rank":         0,
            })
    return paths


# ── forward passes ────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_preserve(model, input_ids, cache, pre_selected,
                     start_layer, end_layer, device, use_bias=True,
                     anti_align=False):
    """Replace last-token MLP output with sign-gated skeleton reconstruction."""
    transformer  = model.transformer
    input_ids    = input_ids.to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    x = transformer.wte(input_ids) + transformer.wpe(position_ids)
    x = transformer.drop(x)

    for layer_idx, block in enumerate(transformer.h):
        attn_out = block.attn(block.ln_1(x))[0]
        x = x + attn_out

        if start_layer <= layer_idx <= end_layer:
            ld   = cache.layer_skeleton(layer_idx)
            V    = ld["V"].to(device)
            bias = ld["bias"] if use_bias else None

            h            = block.mlp.act(block.mlp.c_fc(block.ln_2(x)))
            full_mlp_out = block.mlp.c_proj(h)

            h_last = h[0, -1, :].detach().cpu()
            if anti_align:
                gated = [p for p in pre_selected[layer_idx]
                         if h_last[p["neuron_idx"]].item() * p["static_score"] < 0]
            else:
                gated = [p for p in pre_selected[layer_idx]
                         if h_last[p["neuron_idx"]].item() * p["static_score"] > 0]

            if gated:
                ske_last           = reconstruct_mlp_via_skeleton(h[:, -1:, :], gated, V, bias)
                mlp_out            = full_mlp_out.clone()
                mlp_out[:, -1:, :] = ske_last
            else:
                mlp_out = full_mlp_out
        else:
            mlp_out = block.mlp(block.ln_2(x))

        x = x + mlp_out

    return model.lm_head(transformer.ln_f(x))


@torch.no_grad()
def forward_ablation(model, input_ids, cache, pre_selected,
                     start_layer, end_layer, device, use_bias=True):
    """Subtract full skeleton contribution from last-token MLP output (no sign-gate)."""
    transformer  = model.transformer
    input_ids    = input_ids.to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    x = transformer.wte(input_ids) + transformer.wpe(position_ids)
    x = transformer.drop(x)

    for layer_idx, block in enumerate(transformer.h):
        attn_out = block.attn(block.ln_1(x))[0]
        x = x + attn_out

        if start_layer <= layer_idx <= end_layer:
            ld   = cache.layer_skeleton(layer_idx)
            V    = ld["V"].to(device)
            bias = ld["bias"] if use_bias else None

            h            = block.mlp.act(block.mlp.c_fc(block.ln_2(x)))
            full_mlp_out = block.mlp.c_proj(h)

            top_paths = pre_selected[layer_idx]
            if top_paths:
                ske_last           = reconstruct_mlp_via_skeleton(h[:, -1:, :], top_paths, V, bias)
                mlp_out            = full_mlp_out.clone()
                mlp_out[:, -1:, :] = full_mlp_out[:, -1:, :] - ske_last
            else:
                mlp_out = full_mlp_out
        else:
            mlp_out = block.mlp(block.ln_2(x))

        x = x + mlp_out

    return model.lm_head(transformer.ln_f(x))


# ── experiment runner ─────────────────────────────────────────────────────────

def run_channel_first_experiment(args):
    from block_interp.model_load import load_model_and_embeddings
    from ske_pre_valid.within_ske_cache import WithinSkeCache

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer, W_emb = load_model_and_embeddings(args.model_name, device)
    model.eval()

    from ske_pre_valid.static_skeleton import _resolve_target_info
    target_id = _resolve_target_info(tokenizer, args.target_word)["target_id"]
    cache     = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)
    use_bias  = getattr(args, "use_bias", True)

    print(f"\nBuilding channel-first skeleton"
          f"  (target={args.target_word!r},"
          f"  k_channel={args.k_channel}, k_neuron={args.k_neuron}) ...")

    ch_selected  = {}
    rnd_selected = {}
    for li in range(args.start_layer, args.end_layer + 1):
        ld = cache.layer_skeleton(li)
        ch_selected[li] = select_channel_first(
            ld["delta"], ld["V"], W_emb, target_id,
            args.k_channel, args.k_neuron,
        )
        rnd_selected[li] = select_random_paths(
            ld["delta"], len(ch_selected[li]), seed=args.random_seed + li,
        )
        print(f"  Layer {li:3d}: {len(ch_selected[li])} paths")

    with open(args.dataset_path) as f:
        raw = json.load(f)
    prompts = [raw[k]["prompt"] for k in sorted(raw.keys(), key=int)[:args.n_samples]]
    prompts = [apply_template(p, args.model_name, tokenizer) for p in prompts]
    add_special_tokens = needs_true_add_special_tokens(args.model_name, tokenizer)

    sep   = "═" * 70
    label = (f"Channel-first  k_ch={args.k_channel}  k_neu={args.k_neuron}"
             f"  layers {args.start_layer}–{args.end_layer}")
    print(f"\n{sep}\n  {label}  |  target={args.target_word!r}\n{sep}")

    ori_r, ch_r, abl_r, rnd_r, anti_r = [], [], [], [], []
    ori_p, ch_p, abl_p, rnd_p, anti_p = [], [], [], [], []

    for idx, prompt in enumerate(prompts, 1):
        ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        with torch.no_grad():
            ori_log = model(ids.to(device)).logits[0, -1, :]
        o_r, o_p = target_rank_prob(ori_log, target_id)

        s_log = forward_preserve(
            model, ids, cache, ch_selected,
            args.start_layer, args.end_layer, str(device), use_bias=use_bias,
        )[0, -1, :]
        s_r, s_p = target_rank_prob(s_log, target_id)

        a_log = forward_ablation(
            model, ids, cache, ch_selected,
            args.start_layer, args.end_layer, str(device), use_bias=use_bias,
        )[0, -1, :]
        a_r, a_p = target_rank_prob(a_log, target_id)

        r_log = forward_preserve(
            model, ids, cache, rnd_selected,
            args.start_layer, args.end_layer, str(device), use_bias=use_bias,
        )[0, -1, :]
        rn_r, rn_p = target_rank_prob(r_log, target_id)

        anti_log = forward_preserve(
            model, ids, cache, ch_selected,
            args.start_layer, args.end_layer, str(device), use_bias=use_bias,
            anti_align=True,
        )[0, -1, :]
        anti_r_, anti_p_ = target_rank_prob(anti_log, target_id)

        ori_r.append(o_r);   ch_r.append(s_r);   abl_r.append(a_r)
        rnd_r.append(rn_r);  anti_r.append(anti_r_)
        ori_p.append(o_p);   ch_p.append(s_p);   abl_p.append(a_p)
        rnd_p.append(rn_p);  anti_p.append(anti_p_)

        print(f"\n[{idx:2d}] {prompt!r}")
        print(f"     Original    rank={o_r:5d}  prob={o_p:.5f}")
        print(f"     Ch.preserve rank={s_r:5d}  prob={s_p:.5f}")
        print(f"     Ch.ablation rank={a_r:5d}  prob={a_p:.5f}")
        print(f"     Random      rank={rn_r:5d}  prob={rn_p:.5f}")
        print(f"     Anti-align  rank={anti_r_:5d}  prob={anti_p_:.5f}")
        if getattr(args, "topk_tokens", 0) > 0:
            print(f"     Top-{args.topk_tokens} (channel-first preserve):")
            for rk, (tok, prob) in enumerate(
                top_tokens(s_log, tokenizer, args.topk_tokens), 1
            ):
                marker = " ◄" if tok.strip() == args.target_word.strip() else ""
                print(f"       {rk:3d}.  {tok!r:25s}  {prob:.5f}{marker}")

    n = len(prompts)
    print(f"\n{sep}\n  Summary ({n} prompts)")
    print(f"  {'':22s}  {'MeanRank':>9}  {'MeanProb':>10}  {'Top-5':>7}  {'Top-20':>8}")
    for lbl, ranks, probs in [
        ("Original",       ori_r,  ori_p),
        ("Ch.preserve",    ch_r,   ch_p),
        ("Ch.ablation",    abl_r,  abl_p),
        ("Random",         rnd_r,  rnd_p),
        ("Anti-align.pre", anti_r, anti_p),
    ]:
        print(f"  {lbl:22s}  {sum(ranks)/n:>9.1f}  "
              f"{sum(probs)/n:>10.6f}  "
              f"{sum(r <= 5  for r in ranks):>4}/{n}  "
              f"{sum(r <= 20 for r in ranks):>5}/{n}")
    print(sep)
