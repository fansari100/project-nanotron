"""Pluggable supervised training loop.

Five lines of friction between "I have a torch model" and "I have a
trained torch model": optimizer, scheduler, AMP, gradient clipping,
early stopping.  No external trainer dep.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = True
    early_stop_patience: int = 8
    log_every: int = 50


def supervised_loop(model, train_loader, val_loader, loss_fn, cfg: TrainConfig):
    import torch
    from torch.amp import GradScaler, autocast

    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler(device.type) if cfg.use_amp and device.type == "cuda" else None

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            x, y = (b.to(device) for b in batch)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast(device.type):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            train_loss += float(loss.item())
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = (b.to(device) for b in batch)
                pred = model(x)
                val_loss += float(loss_fn(pred, y).item())

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                break

    return {"best_val": best_val, "history": history}
