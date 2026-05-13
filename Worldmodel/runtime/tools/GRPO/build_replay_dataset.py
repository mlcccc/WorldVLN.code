#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


def _rank01_average_ties(vals: np.ndarray) -> np.ndarray:
    n = int(vals.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.zeros((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks / float(max(1, n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--lambda_act", type=float, default=1.0)
    ap.add_argument("--lambda_task", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.0)
    ap.add_argument("--alpha_decay", type=float, default=0.9)
    ap.add_argument(
        "--mode",
        type=str,
        default="precomputed_adv",
        choices=["precomputed_adv", "raw_reward", "gate_mean", "rank_gate"],
        help="How to derive grpo_weight/grpo_score/grpo_gate from rewards (legacy rank_gate kept for compatibility).",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        gid = str(r.get("grpo_group_id", ""))
        if not gid:
            gid = f"__single_{i}"
        groups.setdefault(gid, []).append(i)

    for _, inds in groups.items():
        idx = np.asarray(inds, dtype=np.int64)
        r_act = np.asarray([float(rows[i].get("grpo_reward_act", rows[i].get("grpo_reward", 0.0))) for i in idx], dtype=np.float64)
        r_task = np.asarray([float(rows[i].get("grpo_reward_task", 0.0)) for i in idx], dtype=np.float64)
        # Prefer CE advantage if present (more GRPO-ish); else fall back to raw CE.
        r_ce = np.asarray(
            [
                float(
                    rows[i].get(
                        "grpo_reward_ce_adv",
                        rows[i].get("grpo_reward_ce_raw", rows[i].get("grpo_reward_ce", 0.0)),
                    )
                )
                for i in idx
            ],
            dtype=np.float64,
        )
        clip_ids = np.asarray([int(rows[i].get("grpo_clip_id", 1)) for i in idx], dtype=np.int64)
        r = float(args.lambda_act) * r_act + float(args.lambda_task) * r_task + float(args.lambda_ce) * r_ce
        mode = str(args.mode or "raw_reward").strip().lower()
        has_precomputed_adv = all("grpo_adv_final" in rows[i] for i in idx.tolist())
        adv_pre = np.asarray([float(rows[i].get("grpo_adv_final", 0.0)) for i in idx], dtype=np.float64)
        if mode == "precomputed_adv" and has_precomputed_adv:
            s = r.copy()
            w = adv_pre.copy()
            m = (w > 0).astype(np.float64)
        elif mode == "gate_mean":
            mu = float(np.mean(r)) if r.size > 0 else 0.0
            m = (r >= mu).astype(np.float64)
            s = r.copy()
            w = m * r
        elif mode == "rank_gate":
            s_act = _rank01_average_ties(r_act)
            s_task = _rank01_average_ties(r_task)
            s_ce = _rank01_average_ties(r_ce)
            s = float(args.lambda_act) * s_act + float(args.lambda_task) * s_task + float(args.lambda_ce) * s_ce
            mu = float(np.mean(r)) if r.size > 0 else 0.0
            m = (r >= mu).astype(np.float64)
            w = (np.power(float(args.alpha_decay), np.maximum(0, clip_ids - 1))) * m * s
        else:
            # raw_reward: no gating, no ranking
            m = np.ones_like(r, dtype=np.float64)
            s = r.copy()
            w = r.copy()
        for j, i in enumerate(idx.tolist()):
            rows[i]["grpo_weight"] = float(w[j])
            rows[i]["grpo_score"] = float(s[j])
            rows[i]["grpo_gate"] = float(m[j])

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_replay_dataset] wrote {len(rows)} lines -> {out_path}")


if __name__ == "__main__":
    main()

