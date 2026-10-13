#!/usr/bin/env python
"""Run a single-sample dynamic Attention-OV + MLP circuit experiment."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_config import within_ske_path
from scripts_ske_cir.run_skeleton_pred import CACHE_BASE, last_layer
from ske_pre_valid.skeleton_pred import CONDITIONS, eval_one_k, setup_skeleton_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint dynamic per-head Attention OV and MLP skeleton prediction"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--in_seq", required=True)
    parser.add_argument("--start_layer", type=int, required=True)
    parser.add_argument("--end_layer", type=int)
    parser.add_argument("--mlp_k", type=int, required=True)
    parser.add_argument("--attention_k", type=int, required=True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--topk_tokens", type=int, default=20)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        default=["ske_pre", "ske_abl"])
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    cache_dir = cli.cache_dir or within_ske_path(cli.model, CACHE_BASE)
    if cache_dir is None:
        raise SystemExit(f"no MLP skeleton cache found for {cli.model}")
    end_layer = cli.end_layer if cli.end_layer is not None else last_layer(cache_dir)
    args = SimpleNamespace(
        model_name=cli.model,
        cache_dir=cache_dir,
        in_seq=cli.in_seq,
        target_word=cli.target,
        start_layer=cli.start_layer,
        end_layer=end_layer,
        k1=cli.mlp_k,
        attention_k1=cli.attention_k,
        attention_selection=None,
        attention_intervention=None,
        topk_tokens=cli.topk_tokens,
        seq_reduce="last",
        random_seed=42,
        use_bias=True,
        rely_ori_acv=False,
        gpu=cli.gpu,
        gemma_restore_magnit=False,
        conditions=cli.conditions,
    )
    setup = setup_skeleton_pred(args)
    result = eval_one_k(
        cli.mlp_k,
        setup,
        args,
        conditions=cli.conditions,
        return_traces=cli.trace,
    )
    stats = result[0]
    print(f"layers={cli.start_layer}-{end_layer} mlp_k={cli.mlp_k} attention_k={cli.attention_k}")
    for condition, values in stats.items():
        print(
            f"{condition:8s} rank={values['rank']:6d} "
            f"logit={values['logit']:+.6f} prob={values['prob']:.6g}"
        )


if __name__ == "__main__":
    main()
