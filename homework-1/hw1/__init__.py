"""Standalone inference prototypes, not integrated TokenSpeed operators."""

from .state_ops import causal_depthwise_conv1d, gated_delta_recurrence

__all__ = ["causal_depthwise_conv1d", "gated_delta_recurrence"]
