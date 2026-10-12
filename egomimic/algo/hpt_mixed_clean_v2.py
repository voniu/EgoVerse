"""Compatibility imports for checkpoints created before the strategy rename."""

from egomimic.algo.hpt_residual_joint import (
    HPTResidualJoint,
    attach_keyed_normalization_metadata,
)

HPTMixedCleanV2 = HPTResidualJoint

__all__ = ["HPTMixedCleanV2", "attach_keyed_normalization_metadata"]
