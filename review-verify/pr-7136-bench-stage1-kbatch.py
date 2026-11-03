"""Decode-shape timing for ROCm/aiter PR #7136: CK-Tile MXFP4 stage 1 at k_batch 2 vs 1.

Shapes are the gpt-oss a16w4 rows of aiter/configs/model_configs/gptoss_a16w4_tuned_fmoe.csv:
D=3072, I in {512, 1536, 3072}, E=128, topk=4, token 1..128, bf16 activations, SwiGLU.
k_batch 1 is what the PR launches when hidden_pad > 0 (hidden_pad 192 -> k_pad_zeros 128).
k_batch 2 is what main launched there (numerically wrong with padding); here it is only a speed reference.

Rows per (I, token):
  gemm1      aiter.moe_cktile2stages_gemm1 alone, split-k kernel, N-wide workspace:
             k_batch 2 / k_pad 0  vs  k_batch 1 / k_pad 128
  stage1     cktile_moe_stage1 (workspace zeroing + gemm1 + SwiGLU post-activation), split_k=2 requested:
             k_pad_zeros 0 (launches k_batch 2)  vs  k_pad_zeros 128 (launches k_batch 1)
  fused_moe  fused_moe end to end with gpt-oss biases: hidden_pad 0  vs  hidden_pad 192
             (the launched k_batch of each is recorded in the kb2/kb1 columns)
Times are device time per call (aiter.test_common.run_perftest, torch profiler); ratio = kb1_us / kb2_us.
Weights are random MXFP4 bytes with random e8m0 scales; values do not matter for timing.

Usage (inside a ROCm container with the PR branch's aiter importable, one GPU visible):
  python pr-7136-bench-stage1-kbatch.py --csv out.csv [--inter 512 1536 3072] [--tokens 1 2 4 ...] [--iters 100]
"""

import argparse
import csv
import inspect
import os

# Same as op_tests/test_moe_cktile_stage1_splitk.py: keep the fused_moe row off FlyDSL.
os.environ["AITER_FLYDSL_FORCE"] = "0"

import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage1, fused_moe, fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest
from aiter.utility.fp4_utils import e8m0_shuffle

D = 3072
E = 128
TOPK = 4
HIDDEN_PAD = 192
K_PAD = HIDDEN_PAD // 128 * 128
SWIGLU = ActivationType.Swiglu


def mxfp4_weight(e, n, k):
    q = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(124, 129, (e * n, k // 32), dtype=torch.uint8, device="cuda")
    return shuffle_weight(q.view(dtypes.fp4x2), (16, 16)), e8m0_shuffle(s).view(
        dtypes.fp8_e8m0
    )


def launched_k_batches(fn):
    gemm1 = aiter.moe_cktile2stages_gemm1
    sig = inspect.signature(gemm1)
    seen = []

    def spy(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(bound.arguments["split_k"])
        return gemm1(*args, **kwargs)

    aiter.moe_cktile2stages_gemm1 = spy
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        aiter.moe_cktile2stages_gemm1 = gemm1
    return seen


def bench(fn, iters):
    _, us = run_perftest(fn, num_iters=iters, num_warmup=10)
    return us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inter", type=int, nargs="+", default=[512, 1536, 3072])
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--csv", default="")
    args = p.parse_args()
    torch.manual_seed(0)
    block_m = 16  # get_2stage_cfgs CK-Tile block_m for token < 2048
    rows = []
    for inter in args.inter:
        N = 2 * inter
        w1, s1 = mxfp4_weight(E, N, D)
        w2, s2 = mxfp4_weight(E, D, inter)
        b1 = torch.randn(E, N, device="cuda").clamp(-1.0, 1.0)
        b2 = torch.randn(E, D, device="cuda").clamp(-1.0, 1.0)
        b2[:, D - HIDDEN_PAD :] = 0
        for token in args.tokens:
            x = torch.randn(token, D, device="cuda", dtype=torch.bfloat16) * 0.1
            x_pad = x.clone()
            x_pad[:, D - HIDDEN_PAD :] = 0
            score = torch.randn(token, E, device="cuda", dtype=torch.bfloat16)
            tw, tid = fused_topk(x, score, TOPK, True)
            sorted_ids, _, sorted_eids, nvalid, _ = moe_sorting(
                tid, tw, E, D, torch.bfloat16, block_m
            )

            # gemm1 alone
            y = torch.zeros(token * TOPK, N, dtype=torch.bfloat16, device="cuda")

            def gemm1(k_batch, k_pad):
                return lambda: aiter.moe_cktile2stages_gemm1(
                    x,
                    w1,
                    y,
                    sorted_ids,
                    sorted_eids,
                    nvalid,
                    TOPK,
                    k_padded_zeros=k_pad,
                    w_scale=s1,
                    activation=SWIGLU,
                    block_m=block_m,
                    split_k=k_batch,
                )

            kb2 = bench(gemm1(2, 0), args.iters)
            kb1 = bench(gemm1(1, K_PAD), args.iters)
            rows.append(dict(inter=inter, token=token, row="gemm1", kb2_launched=2, kb1_launched=1, kb2_us=kb2, kb1_us=kb1))

            # cktile_moe_stage1 (zeros + gemm1 + post-activation)
            out = torch.empty(token, TOPK, inter, dtype=torch.bfloat16, device="cuda")

            def stage1(k_pad):
                return lambda: cktile_moe_stage1(
                    x,
                    w1,
                    w2,
                    sorted_ids,
                    sorted_eids,
                    nvalid,
                    out,
                    TOPK,
                    block_m,
                    None,
                    s1,
                    k_pad_zeros=k_pad,
                    activation=SWIGLU,
                    split_k=2,
                    dtype=torch.bfloat16,
                    post_activation_layout="standard",
                )

            l2 = launched_k_batches(stage1(0))
            l1 = launched_k_batches(stage1(K_PAD))
            kb2 = bench(stage1(0), args.iters)
            kb1 = bench(stage1(K_PAD), args.iters)
            rows.append(dict(inter=inter, token=token, row="stage1", kb2_launched=l2, kb1_launched=l1, kb2_us=kb2, kb1_us=kb1))

            # fused_moe end to end with biases
            def e2e(xx, hp):
                return lambda: fused_moe(
                    xx,
                    w1,
                    w2,
                    tw,
                    tid,
                    activation=SWIGLU,
                    quant_type=QuantType.per_1x32,
                    w1_scale=s1,
                    w2_scale=s2,
                    hidden_pad=hp,
                    bias1=b1,
                    bias2=b2,
                )

            fm.get_2stage_cfgs.cache_clear()
            l2 = launched_k_batches(e2e(x, 0))
            l1 = launched_k_batches(e2e(x_pad, HIDDEN_PAD))
            kb2 = bench(e2e(x, 0), args.iters)
            kb1 = bench(e2e(x_pad, HIDDEN_PAD), args.iters)
            rows.append(dict(inter=inter, token=token, row="fused_moe", kb2_launched=l2, kb1_launched=l1, kb2_us=kb2, kb1_us=kb1))

            for r in rows[-3:]:
                r["ratio"] = r["kb1_us"] / r["kb2_us"]
                print(
                    f"I={inter:5d} token={token:4d} {r['row']:9s} "
                    f"kb2 {r['kb2_us']:8.2f} us (launched {r['kb2_launched']})  "
                    f"kb1 {r['kb1_us']:8.2f} us (launched {r['kb1_launched']})  "
                    f"ratio {r['ratio']:.3f}",
                    flush=True,
                )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
