"""Utility helpers for the Fusion framework."""

def count_parameters(model) -> int:
    """Return the total number of parameters in this model."""
    return sum(p.numel() for p in model.parameters())
