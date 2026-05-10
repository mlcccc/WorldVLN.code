"""
Local datasets package for TSformer-VO.

Note: This repository uses a top-level folder named `datasets`, which can clash
with the HuggingFace `datasets` package when running in some Python envs.
Creating this file makes the local folder an explicit package, and our training
script also prepends the repo root to sys.path to ensure local imports win.
"""

