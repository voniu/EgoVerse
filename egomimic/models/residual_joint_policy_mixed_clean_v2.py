"""Compatibility import for checkpoints created before the strategy rename."""

from egomimic.models.residual_joint_transformer_keyed import (
    KeyedResidualJointTransformer,
)

ResidualJointPolicyMixedCleanV2 = KeyedResidualJointTransformer

__all__ = ["ResidualJointPolicyMixedCleanV2"]
