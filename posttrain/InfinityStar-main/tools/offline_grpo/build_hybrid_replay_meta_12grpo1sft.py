#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Item:
    base: str
    clip_id: int
    cand_id: int
    obj: dict


def _traj_base(traj_id: str) -> str:
    # Expected: 000000_k00_c1
    s = str(traj_id)
    if "_k" in s:
        return s.split("_k", 1)[0]
    return s


def _read_all_parts(inp_dir: str) -> List[dict]:
    parts = []
    for name in sorted(os.listdir(inp_dir)):
        if not (name.endswith(".jsonl") and name.startswith("part_")):
            continue
        parts.append(os.path.join(inp_dir, name))
    if not parts:
        raise FileNotFoundError(f"no part_*.jsonl under: {inp_dir}")
    rows: List[dict] = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _index_rows(rows: List[dict]) -> Dict[str, Dict[Tuple[int, int], dict]]:
    # base -> (clip_id, cand_id) -> obj
    out: Dict[str, Dict[Tuple[int, int], dict]] = defaultdict(dict)
    for obj in rows:
        traj_id = str(obj.get("traj_id", "") or "")
        if not traj_id:
            continue
        base = _traj_base(traj_id)
        clip_id = int(obj.get("grpo_clip_id", 0) or 0)
        cand_id = int(obj.get("candidate_id", -1) if obj.get("candidate_id", None) is not None else -1)
        if cand_id < 0:
            # fallback: parse _kXX
            if "_k" in traj_id:
                try:
                    cand_id = int(traj_id.split("_k", 1)[1].split("_", 1)[0])
                except Exception:
                    cand_id = -1
        if clip_id <= 0 or cand_id < 0:
            continue
        out[base][(clip_id, cand_id)] = obj
    return out


def _make_anchor(example_obj: dict, *, base: str, anchor_begin: int, anchor_end: int, fps: int) -> dict:
    a = dict(example_obj)
    a["begin_frame_id"] = int(anchor_begin)
    a["end_frame_id"] = int(anchor_end)
    a["fps"] = int(fps)
    a["traj_id"] = f"{base}_sft"
    a["hybrid_role"] = "sft"
    # Ensure GRPO-specific fields won't accidentally be used.
    a["grpo_reward"] = 0.0
    a["grpo_old_logprob"] = 0.0
    a["grpo_ref_logprob"] = 0.0
    a["grpo_trace_files"] = []
    a["grpo_group_id"] = str(a.get("grpo_group_id", base))
    a["grpo_clip_id"] = 0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_replay_meta_dir", required=True)
    ap.add_argument("--output_replay_meta_dir", required=True)
    ap.add_argument("--num_parts", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="candidates per clip")
    ap.add_argument("--num_clips", type=int, default=3)
    ap.add_argument("--add_sft_anchor", type=int, default=1)
    ap.add_argument("--anchor_begin", type=int, default=0)
    ap.add_argument("--anchor_end", type=int, default=48)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--pad_to_equal_groups", type=int, default=1, help="duplicate early groups to equalize shard lengths")
    args = ap.parse_args()

    rows = _read_all_parts(args.input_replay_meta_dir)
    idx = _index_rows(rows)
    bases = sorted(idx.keys())
    if not bases:
        raise RuntimeError("no valid traj_id/grpo_clip_id/candidate_id rows found")

    groups: List[List[dict]] = []
    missing = 0
    for base in bases:
        m = idx[base]
        ok = True
        lst: List[dict] = []
        for c in range(1, int(args.num_clips) + 1):
            for k in range(int(args.k)):
                obj = m.get((c, k), None)
                if obj is None:
                    ok = False
                    break
                o2 = dict(obj)
                o2["hybrid_role"] = "grpo"
                lst.append(o2)
            if not ok:
                break
        if not ok:
            missing += 1
            continue
        if int(args.add_sft_anchor) == 1:
            anchor = _make_anchor(lst[0], base=base, anchor_begin=args.anchor_begin, anchor_end=args.anchor_end, fps=args.fps)
            lst.append(anchor)
        groups.append(lst)

    if not groups:
        raise RuntimeError("no complete groups built (check k/num_clips)")

    os.makedirs(args.output_replay_meta_dir, exist_ok=True)

    # Shard by group index modulo num_parts.
    shards: List[List[List[dict]]] = [[] for _ in range(int(args.num_parts))]
    for gi, g in enumerate(groups):
        shards[gi % int(args.num_parts)].append(g)

    # Optionally pad to equal group count to avoid dropping unique groups due to min-iters across ranks.
    lens = [len(s) for s in shards]
    max_groups = max(lens)
    if int(args.pad_to_equal_groups) == 1 and max_groups > 0:
        # Pick a non-empty source pool for padding when a shard is empty (rare).
        pad_pool: List[List[dict]] = []
        for s in shards:
            if len(s) > 0:
                pad_pool = s
                break
        for pi in range(int(args.num_parts)):
            while len(shards[pi]) < max_groups:
                # duplicate from the beginning (stable)
                if len(shards[pi]) > 0:
                    shards[pi].append(shards[pi][0])
                elif pad_pool:
                    shards[pi].append(pad_pool[len(shards[pi]) % len(pad_pool)])
                else:
                    break

    # Write part files: preserve order (group-major; within group: 12 grpo then 1 sft).
    total_out = 0
    for pi in range(int(args.num_parts)):
        out_p = os.path.join(args.output_replay_meta_dir, f"part_{pi:02d}.jsonl")
        with open(out_p, "w", encoding="utf-8") as f:
            for g in shards[pi]:
                for obj in g:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_out += 1

    print(
        json.dumps(
            {
                "input_dir": args.input_replay_meta_dir,
                "output_dir": args.output_replay_meta_dir,
                "bases_total": len(bases),
                "groups_complete": len(groups),
                "groups_missing": int(missing),
                "num_parts": int(args.num_parts),
                "groups_per_part": [len(s) for s in shards],
                "rows_out": int(total_out),
                "rows_per_group": len(groups[0]) if groups else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

