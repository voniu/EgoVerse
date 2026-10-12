"""Compatibility import for checkpoints created before the strategy rename."""

from egomimic.models.residual_joint_transformer import ResidualJointTransformer

ResidualJointPolicyCleanV2 = ResidualJointTransformer

__all__ = ["ResidualJointPolicyCleanV2"]
