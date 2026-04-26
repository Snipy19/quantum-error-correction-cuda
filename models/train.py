import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import pickle, os, time, json, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.model import QuantumNet, QuantumNetDeep

EPOCHS     = 400
LR         = 1e-3
BATCH_SIZE = 128
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset.pkl")
SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))
device     = "cuda" if torch.cuda.is_available() else "cpu"


def fidelity(pred, target):
    return (torch.sum(torch.sqrt(torch.clamp(pred * target, 1e-10)), dim=-1) ** 2).mean()


def loss_fn(pred, target):
    f   = 1.0 - fidelity(pred, target)
    mse = F.mse_loss(pred, target)
    kl  = F.kl_div(torch.log(pred + 1e-8), target, reduction='batchmean')
    return 0.6 * f + 0.3 * mse + 0.1 * kl


def cosine_lr(opt, epoch, total, warmup, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        p  = (epoch - warmup) / max(1, total - warmup)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * p))
    for pg in opt.param_groups:
        pg['lr'] = lr
    return lr


def train_model(model, X, y, tag):
    model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = GradScaler('cuda') if device == "cuda" else None
    n      = len(X)

    print(f"\n{'='*55}")
    print(f"  Training: {tag}  |  device={device}  |  n={n}")
    print(f"{'='*55}")

    best_loss, best_fid, best_state = 999, 0, None
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        lr  = cosine_lr(opt, epoch, EPOCHS, 30, LR)
        idx = torch.randperm(n)
        Xs, ys = X[idx], y[idx]
        ep_loss, ep_fid, steps = 0, 0, 0

        for i in range(0, n, BATCH_SIZE):
            xb = Xs[i:i+BATCH_SIZE].to(device)
            yb = ys[i:i+BATCH_SIZE].to(device)
            opt.zero_grad(set_to_none=True)

            if scaler:
                with autocast('cuda'):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            with torch.no_grad():
                ep_fid += fidelity(pred.detach(), yb).item()
            ep_loss += loss.item(); steps += 1

        avg_loss = ep_loss / steps
        avg_fid  = ep_fid  / steps

        if avg_fid > best_fid:
            best_fid   = avg_fid
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == EPOCHS-1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.5f} | fidelity={avg_fid:.5f} | lr={lr:.5f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    path = os.path.join(SAVE_DIR, f"{tag}.pth")
    torch.save(model.state_dict(), path)
    print(f"  ✓ Best fidelity={best_fid:.5f} → {path}")
    return model


def main():
    print(f"[train] Loading {DATA_PATH}")
    data = pickle.load(open(DATA_PATH, "rb"))
    X = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    y = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
    X = X / X.sum(-1, keepdim=True).clamp(1e-8)
    y = y / y.sum(-1, keepdim=True).clamp(1e-8)
    print(f"[train] {len(X)} samples | device={device}")

    train_model(QuantumNet(),     X, y, "model")
    train_model(QuantumNetDeep(), X, y, "model_deep")
    print("\n[train] Done! Run app.py")


if __name__ == "__main__":
    main()
