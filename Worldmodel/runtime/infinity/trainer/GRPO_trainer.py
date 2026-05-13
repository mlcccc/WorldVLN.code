#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO trainer.

This class reuses the stable InfinityTrainer training core and enables
reward-weighted objective via `args.trainer_type=grpo`.
"""

from infinity.trainer.sft_trainer import InfinityTrainer


class GRPOTrainer(InfinityTrainer):
    pass

