"""Minimal baseline trainer for the weld-defect CustomCNN.

Trains a from-scratch CustomCNN on the synthetic radiographs using only the
package's own PreprocessPipeline + save_model (no MLflow / albumentations / timm
needed). Produces models/ensemble_best.pt in the exact format load_model expects,
so the FastAPI /predict endpoint can serve it.

This is a synthetic smoke-test baseline (TRL-5 status: synthetic data only),
NOT a validated industrial detector.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from weld_defect.data.preprocess import PreprocessPipeline
from weld_defect.models.custom_cnn import CustomCNN
from weld_defect.models.factory import save_model

CLASS_NAMES = ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
DATA_DIR = Path("data/synthetic")
IMG_DIR = DATA_DIR / "images"
CKPT = Path("models/ensemble_best.pt")
INPUT_SIZE = 224
NUM_CLASSES = 6
EPOCHS = 12
BATCH = 32
LR = 1e-3
SEED = 42


class WeldCSVDataset(Dataset):
    def __init__(self, csv_path: Path, pre: PreprocessPipeline):
        self.df = pd.read_csv(csv_path)
        self.pre = pre

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        arr = self.pre(IMG_DIR / row["image"])  # HWC float, normalized, 3 channels
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()
        y = int(row["label_idx"])
        return x, y


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pre = PreprocessPipeline(input_size=INPUT_SIZE)
    train_ds = WeldCSVDataset(DATA_DIR / "train.csv", pre)
    val_ds = WeldCSVDataset(DATA_DIR / "val.csv", pre)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # class-balanced weights (no_defect is majority)
    counts = train_ds.df["label_idx"].value_counts().sort_index().to_numpy()
    weights = torch.tensor(len(train_ds.df) / (NUM_CLASSES * counts), dtype=torch.float32)
    print("class counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    print("class weights:", [round(w, 2) for w in weights.tolist()])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = CustomCNN(num_classes=NUM_CLASSES, input_size=INPUT_SIZE, pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CustomCNN params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=weights)

    device = "cpu"
    model.to(device)
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()
        train_acc = correct / total

        # validation
        model.eval()
        v_correct, v_total = 0, 0
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
                all_p.extend(pred.cpu().tolist())
                all_y.extend(y.cpu().tolist())
        val_acc = v_correct / v_total
        # macro F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_y, all_p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}/{EPOCHS}  loss={loss_sum/total:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  ({dt:.1f}s)")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    meta = {
        "architecture": "custom_cnn",
        "num_classes": NUM_CLASSES,
        "input_size": INPUT_SIZE,
        "class_names": CLASS_NAMES,
        "kind": "single",
        "trained_on": "synthetic smoke data (data/synthetic)",
        "best_val_f1_macro": float(best_f1),
        "trl_status": "synthetic smoke baseline — not a validated industrial detector",
    }
    save_model(model, CKPT, meta=meta)
    print(f"\n✅ saved checkpoint → {CKPT}  (best val_f1_macro={best_f1:.3f})")
    print(f"   meta.architecture={meta['architecture']}  class_names={meta['class_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
