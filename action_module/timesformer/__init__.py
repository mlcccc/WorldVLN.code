# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
TimeSformer package init.

This repo originally calls `setup_environment()` on import, which depends on `fvcore`.
For lightweight scripts that only need model definitions (e.g. `timesformer.models.vit`)
we allow importing without `fvcore` installed.
"""

try:
    from timesformer.utils.env import setup_environment

    setup_environment()
except ModuleNotFoundError:
    # Allow minimal imports (e.g. VisionTransformer) without optional deps.
    pass
