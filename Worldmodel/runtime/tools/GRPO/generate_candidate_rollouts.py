#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build K-candidate rollout jsonl from task jsonl.

This script prepares grouped offline replay records and seed schedule.
Actual video generation / trace collection can be executed by your rollout runner,
which should write back:
  - traj_id directory with trajectory.json
  - grpo_old_logprob (if available)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _candidate_seed(seed_base: int, task_idx: int, cand_idx: int, task_seed_stride: int, candidate_seed_stride: int) -> int:
    # Keep group ids stable but separate candidate seeds by a large stride so K-candidates
    # explore meaningfully different sampling paths instead of near-identical adjacent seeds.
    return int(seed_base + task_idx * task_seed_stride + cand_idx * candidate_seed_stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=20260320)
    ap.add_argument("--task_seed_stride", type=int, default=1000003)
    ap.add_argument("--candidate_seed_stride", type=int, default=65537)
    args = ap.parse_args()

    task_path = os.path.abspath(args.task_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    with open(task_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    rows: List[Dict[str, Any]] = []
    for ti, t in enumerate(tasks):
        gid = str(t.get("grpo_group_id", f"task_{ti}"))
        clip_id = int(t.get("grpo_clip_id", 1))
        for ki in range(int(args.k)):
            r = dict(t)
            r["grpo_group_id"] = gid
            r["grpo_clip_id"] = int(clip_id)
            r["candidate_id"] = int(ki)
            r["seed"] = _candidate_seed(
                seed_base=int(args.seed_base),
                task_idx=int(ti),
                cand_idx=int(ki),
                task_seed_stride=int(args.task_seed_stride),
                candidate_seed_stride=int(args.candidate_seed_stride),
            )
            r["traj_id"] = f"{ti:06d}_k{ki:02d}"
            r["grpo_old_logprob"] = float(r.get("grpo_old_logprob", 0.0))
            rows.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate_candidate_rollouts] tasks={len(tasks)} k={args.k} rows={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()

