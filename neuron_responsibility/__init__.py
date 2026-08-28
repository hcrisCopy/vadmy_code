"""Baseline-agnostic CLIP-neuron responsibility learning for WS-VAD.

The package initializer intentionally avoids importing PyTorch so the offline
selection and feature-building stages can run in a lightweight NumPy process.
"""
