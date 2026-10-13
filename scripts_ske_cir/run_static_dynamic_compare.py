"""
python scripts_ske_cir/run_static_dynamic_compare.py \
    --model_name gpt2-medium --cache_dir result/within_ske_cache \
    --in_seq "The cat looks very" --target_word " happy" \
    --start_layer 5 --end_layer 22 \
    --k 4000 --k1 200 --gpu 0 --norm_eff

Given a specific input sequence, compare:
  target_alignment[i] = V[i] @ W_emb[target_id]
  static  score[j,i] = |delta[j,i]| * |target_alignment[i]| (input-independent)
  dynamic score[j,i] = h[j] * delta[j,i] * target_alignment[i]
                                                            (input-conditioned, signed)

Per layer reports:
  - Rank of each static top-k path in the full dynamic ordering
  - Jaccard / Recall / Precision between static top-k and dynamic top-k1
  - Average percentile of static paths in dynamic ranking
"""

import os, sys, argparse, statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_interp.model_load import load_model_and_embeddings
from ske_pre_valid.within_ske_cache import WithinSkeCache
from ske_pre_valid.static_skeleton import _effector_scores
from model_config.templates import apply_template, needs_true_add_special_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default="gpt2-medium")
    p.add_argument("--cache_dir",    default="result/within_ske_cache")
    p.add_argument("--in_seq",       default="The cat looks very")
    p.add_argument("--target_word",  default=" happy")
    p.add_argument("--start_layer",  type=int, default=5)
    p.add_argument("--end_layer",    type=int, default=22)
    p.add_argument("--k",            type=int, default=4000,
                   help="Static top-k paths per layer")
    p.add_argument("--k1",           type=int, default=200,
                   help="Dynamic top-k1 paths per layer to compare against")
    p.add_argument("--seq_reduce",   default="last", choices=["last", "mean"])
    p.add_argument("--norm_eff",         action="store_true",
                   help="Globally z-score normalise effector scores across all layers")
    p.add_argument("--gpu",              type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def collect_h_values(model, input_ids, start_layer, end_layer, device, seq_reduce="last"):
    """
    Standard forward pass (no intervention), capturing post-activation h at each target layer.
    h[j] = act(c_fc(ln_2(x)))[j]  — the neuron activation used in dynamic scoring.
    """
    tr = model.transformer
    input_ids = input_ids.to(device)
    x = tr.wte(input_ids) + tr.wpe(
        torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    )
    x = tr.drop(x)

    h_by_layer = {}
    for li, block in enumerate(tr.h):
        x = x + block.attn(block.ln_1(x))[0]
        if start_layer <= li <= end_layer:
            h = block.mlp.act(block.mlp.c_fc(block.ln_2(x)))   # [1, seq, d_ff]
            h_by_layer[li] = (
                h[0, -1, :] if seq_reduce == "last" else h[0].mean(0)
            ).cpu()
            x = x + block.mlp.c_proj(h)
        else:
            x = x + block.mlp(block.ln_2(x))

    return h_by_layer


def build_effectors(cache, W_emb, target_id, start_layer, end_layer,
                    norm_eff):
    """Compute scalar effector-to-target alignments for all layers.

    Each returned value is ``V[i] @ W_emb[target_id]``.  It already contains
    target alignment and must not be multiplied by that alignment a second time.
    """
    raw = {
        li: _effector_scores(
            cache.layer_skeleton(li)["V"], W_emb, target_id,
            cache.layer_skeleton(li).get("post_ffn_norm_weight"),
        )
        for li in range(start_layer, end_layer + 1)
    }
    if not norm_eff:
        return raw
    all_eff = torch.cat(list(raw.values()))
    g_mean  = all_eff.mean()
    g_std   = all_eff.std().clamp(min=1e-8)
    print(f"  [norm_eff] global effector  mean={g_mean:.5f}  std={g_std:.5f}")
    return {li: (raw[li] - g_mean) / g_std for li in raw}


def compare_layer(delta, effector, h_reduced, k, k1):
    """
    Compute per-layer comparison statistics.

    target_alignment[i] = effector[i] = V[i] @ W_emb[target_id]
    static  score[j,i] = |delta[j,i]| * |target_alignment[i]|
    dynamic score[j,i] = h[j] * delta[j,i] * target_alignment[i]

    Static top-k ranks by the non-negative score and Dynamic top-k ranks by signed score,
                          matching within_ske_selector.select_and_reconstruct_skeleton's
                          real selection criterion (not |score|)

    Returns a dict with Jaccard, Recall, Precision, and dynamic-rank statistics
    for the static top-k paths.
    """
    delta    = delta.cpu().float()
    target_alignment = effector.cpu().float()
    h        = h_reduced.cpu().float()

    d_ff, n_sp = delta.shape
    total      = d_ff * n_sp

    signed_static_base = delta * target_alignment.unsqueeze(0)  # [d_ff, n_sp]
    sta_score = delta.abs() * target_alignment.abs().unsqueeze(0)
    dyn_score = h.unsqueeze(1) * signed_static_base

    k_sta = min(k,  total)
    k_dyn = min(k1, total)

    # Static side matches select_global_abs_topk / select_per_neuron_abs_topk, which
    # rank by |delta*effector| — abs is the real static selection criterion.
    sta_flat_abs = sta_score.flatten()
    # Dynamic side must match within_ske_selector.select_and_reconstruct_skeleton,
    # which ranks by the *signed* score h[j]*delta[j,i]*effector[i] (torch.topk with
    # largest=True), not |score| — using abs here would compare against a selection
    # criterion the real dynamic skeleton forward never actually uses.
    dyn_flat = dyn_score.flatten()

    sta_top_set = {
        (idx.item() // n_sp, idx.item() % n_sp)
        for idx in torch.topk(sta_flat_abs, k_sta).indices
    }
    dyn_top_set = {
        (idx.item() // n_sp, idx.item() % n_sp)
        for idx in torch.topk(dyn_flat, k_dyn).indices
    }

    # rank of every path in dynamic ordering (rank 1 = highest signed dynamic score)
    dyn_rank_flat = torch.empty(total, dtype=torch.long)
    dyn_rank_flat[torch.argsort(dyn_flat, descending=True)] = torch.arange(total)

    # dynamic ranks of the static top-k paths
    sta_dyn_ranks = [
        dyn_rank_flat[j * n_sp + i].item() + 1   # 1-indexed
        for j, i in sta_top_set
    ]

    overlap = len(sta_top_set & dyn_top_set)
    union   = len(sta_top_set | dyn_top_set)

    # random baseline: uniformly sample k_sta paths, compute expected overlap with dyn_top_set
    rnd_overlap_expected = k_sta * k_dyn / total

    return {
        "jaccard":              overlap / union if union else 0.0,
        "recall":               overlap / k_sta,
        "precision":            overlap / k_dyn,
        "rnd_recall_expected":  rnd_overlap_expected / k_sta,   # = k_dyn / total
        "dyn_rank_mean":        statistics.mean(sta_dyn_ranks),
        "dyn_rank_median":      statistics.median(sta_dyn_ranks),
        "dyn_rank_max":         max(sta_dyn_ranks),
        "dyn_rank_pct":         statistics.mean(r / total * 100 for r in sta_dyn_ranks),
        "n_sta":   k_sta,
        "n_dyn":   k_dyn,
        "n_total": total,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, W_emb = load_model_and_embeddings(args.model_name, device)
    model.eval()
    cache     = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)
    from ske_pre_valid.static_skeleton import _resolve_target_info
    _tgt_info = _resolve_target_info(tokenizer, args.target_word)
    target_id = _tgt_info["target_id"]
    args.in_seq = apply_template(args.in_seq, args.model_name, tokenizer)
    input_ids = tokenizer.encode(
        args.in_seq,
        return_tensors="pt",
        add_special_tokens=needs_true_add_special_tokens(args.model_name, tokenizer),
    )

    eff_label = "raw-dot" + ("+global-norm" if args.norm_eff else "")

    print(f"\nInput     : {args.in_seq!r}")
    print(f"Target    : {args.target_word!r}  (id={target_id}, decoded={_tgt_info['target_decoded']!r})")
    print(f"Layers    : [{args.start_layer}, {args.end_layer}]")
    print(f"k_static  : {args.k}   k1_dynamic : {args.k1}")
    print(f"Effector  : {eff_label}\n")

    effectors  = build_effectors(cache, W_emb.cpu(), target_id,
                                 args.start_layer, args.end_layer,
                                 args.norm_eff)
    h_by_layer = collect_h_values(model, input_ids,
                                  args.start_layer, args.end_layer,
                                  str(device), args.seq_reduce)

    sep  = "─" * 108
    hdr  = (f"  {'Layer':>5}  {'Jaccard':>8}  {'Recall':>7}  {'RndRecall':>10}  {'Precis':>7}"
            f"  {'MeanDynRank':>12}  {'MedianDynRank':>13}  {'MaxDynRank':>10}  {'AvgPct%':>8}")
    print(sep)
    print(hdr)
    print(sep)

    layer_stats = {}
    for li in range(args.start_layer, args.end_layer + 1):
        ld    = cache.layer_skeleton(li)
        stats = compare_layer(ld["delta"], effectors[li], h_by_layer[li],
                              args.k, args.k1)
        layer_stats[li] = stats
        print(f"  {li:>5}  {stats['jaccard']:>8.4f}  {stats['recall']:>7.4f}"
              f"  {stats['rnd_recall_expected']:>10.6f}  {stats['precision']:>7.4f}"
              f"  {stats['dyn_rank_mean']:>12.1f}  {stats['dyn_rank_median']:>13.1f}"
              f"  {stats['dyn_rank_max']:>10d}  {stats['dyn_rank_pct']:>8.3f}")

    print(sep)

    n = len(layer_stats)
    def avg(key): return sum(s[key] for s in layer_stats.values()) / n

    print(f"\n  Summary across {n} layers  [k_static={args.k}  k1_dynamic={args.k1}]")
    print(f"    Jaccard  (|sta∩dyn| / |sta∪dyn|)            : {avg('jaccard'):.4f}")
    print(f"    Recall   (static top-k found in dyn top-k1)  : {avg('recall'):.4f}")
    print(f"    Recall   (random baseline expected)           : {avg('rnd_recall_expected'):.6f}")
    print(f"    Lift over random                              : {avg('recall')/avg('rnd_recall_expected'):.1f}x")
    print(f"    Precision(dyn top-k1 covered by static top-k) : {avg('precision'):.4f}")
    print(f"    Mean dynamic rank of static paths             : {avg('dyn_rank_mean'):.1f}")
    print(f"    Median dynamic rank of static paths           : {avg('dyn_rank_median'):.1f}")
    print(f"    Avg percentile in dynamic ranking             : {avg('dyn_rank_pct'):.3f}%")
    print(f"      (0% = top of dynamic ranking, 100% = bottom)")
    print(sep)


if __name__ == "__main__":
    main()
