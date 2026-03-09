"""Pluggable loss functions for CortexLab V2."""

from cortexlab.losses.sft import SFTLoss
from cortexlab.losses.dpo import DPOLoss

__all__ = ["SFTLoss", "DPOLoss"]
