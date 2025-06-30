"""Generic supervised + self-supervised training loops."""

from .loop import TrainConfig, supervised_loop

__all__ = ["TrainConfig", "supervised_loop"]
