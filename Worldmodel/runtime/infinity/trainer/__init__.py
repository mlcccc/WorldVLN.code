# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_trainer(args):
    trainer_type = str(getattr(args, "trainer_type", "sft") or "sft").strip().lower()
    if trainer_type == "grpo":
        from infinity.trainer.GRPO_trainer import GRPOTrainer as Trainer
    else:
        from infinity.trainer.sft_trainer import InfinityTrainer as Trainer
    return Trainer