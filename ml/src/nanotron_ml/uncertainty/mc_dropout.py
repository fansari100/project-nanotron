"""Monte-Carlo Dropout predictor.

Wraps a torch model that contains ``nn.Dropout`` layers and turns it
into a Bayesian predictor by leaving dropout *on* at inference and
averaging over T forward passes.  Returns mean + epistemic-stddev.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCDropoutPredictor:
    """Wrap a torch model for MC Dropout inference."""

    n_samples: int = 32

    def predict(self, model, x):
        import torch

        model.eval()
        # Re-enable dropout layers (BN, etc. stay in eval mode).
        for module in model.modules():
            if module.__class__.__name__.lower().startswith("dropout"):
                module.train()
        outs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                outs.append(model(x))
        stack = torch.stack(outs, dim=0)
        return stack.mean(dim=0), stack.std(dim=0)
