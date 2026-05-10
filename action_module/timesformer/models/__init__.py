# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
timesformer.models package init.

The upstream project imports build utilities that depend on `fvcore`.
For lightweight usage (e.g. importing `timesformer.models.vit.VisionTransformer`)
we allow importing without `fvcore` installed.
"""

try:
    from .build import MODEL_REGISTRY, build_model  # noqa
    from .custom_video_model_builder import *  # noqa
    from .video_model_builder import ResNet, SlowFast  # noqa
except ModuleNotFoundError:
    # Allow minimal imports without optional deps.
    pass
