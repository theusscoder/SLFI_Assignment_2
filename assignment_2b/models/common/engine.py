"""Training / evaluation loop shared by the speech, text and fusion pipelines."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models.common.data import EMOTIONS, ROOT, make_split
from models.common.features import load_ragged
from models.common.nets import FusionModel, SpeechEncoder, TextEncoder, UnimodalModel

import pandas as pd

RUNS = ROOT / "runs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pad(batch):
    lens = [len(b) for b in batch]
    x = torch.zeros(len(batch), max(lens), batch[0].shape[1])
    for i, b in enumerate(batch):
        x[i, :len(b)] = torch.from_numpy(b.astype(np.float32))
    mask = torch.arange(max(lens))[None] < torch.tensor(lens)[:, None]
    return x.to(DEV), mask.to(DEV)


class Task:
    """Wraps model construction and batching for one modality setting."""

    def __init__(self, modality, speech_feat="wavlm", fusion="gated"):
        self.modality, self.speech_feat, self.fusion = modality, speech_feat, fusion
        self.meta = pd.read_csv(ROOT / "cache" / "metadata.csv")
        self.y = self.meta.label.values
        self.S = load_ragged(speech_feat) if modality in ("speech", "fusion") else None
        self.T = load_ragged("bert") if modality in ("text", "fusion") else None

    def name(self, split, seed):
        tag = {"speech": f"speech_{self.speech_feat}", "text": "text_bert",
               "fusion": f"fusion_{self.fusion}_{self.speech_feat}"}[self.modality]
        return f"{tag}/{split}/seed{seed}"

    def build(self):
        if self.modality == "speech":
            m = UnimodalModel(SpeechEncoder(self.S[0].shape[1]))
        elif self.modality == "text":
            m = UnimodalModel(TextEncoder(self.T[0].shape[1]))
        else:
            m = FusionModel(self.S[0].shape[1], self.T[0].shape[1], self.fusion)
        return m.to(DEV)

    def forward(self, model, idx, train=False):
        if self.modality == "speech":
            S = [self.S[i] for i in idx]
            if train:  # random crop as light augmentation for the temporal model
                S = [s[np.random.randint(0, max(1, len(s) // 5)):] for s in S]
            return model(*pad(S))
        if self.modality == "text":
            return model(*pad([self.T[i] for i in idx]))
        S = [self.S[i] for i in idx]
        if train:
            S = [s[np.random.randint(0, max(1, len(s) // 5)):] for s in S]
        return model(*pad(S), *pad([self.T[i] for i in idx]))


@torch.no_grad()
def predict(task, model, idx, bs=64):
    model.eval()
    logits, embs = [], {}
    for b in range(0, len(idx), bs):
        lo, emb = task.forward(model, idx[b:b + bs])
        logits.append(lo.cpu())
        for k, v in emb.items():
            embs.setdefault(k, []).append(v.cpu())
    return torch.cat(logits).numpy(), {k: torch.cat(v).numpy() for k, v in embs.items()}


def train(task, split, seed, epochs=40, bs=32, lr=3e-4, patience=8):
    set_seed(seed)
    tr, va, te = make_split(task.meta, split, seed)
    model = task.build()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + bs - 1) // bs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    y = torch.tensor(task.y, device=DEV)
    out = RUNS / task.name(split, seed)
    out.mkdir(parents=True, exist_ok=True)
    best, bad, hist = -1, 0, []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            logits, _ = task.forward(model, idx, train=True)
            loss = loss_fn(logits, y[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        va_acc = accuracy_score(task.y[va], predict(task, model, va)[0].argmax(1))
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_acc": va_acc})
        if va_acc > best:
            best, bad = va_acc, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"{task.name(split, seed)}: best val acc {best:.3f} after {len(hist)} epochs", flush=True)
    return out


def test(task, split, seed):
    out = RUNS / task.name(split, seed)
    _, _, te = make_split(task.meta, split, seed)
    model = task.build()
    model.load_state_dict(torch.load(out / "model.pt", map_location=DEV))
    logits, embs = predict(task, model, te)
    pred = logits.argmax(1)
    yt = task.y[te]
    res = {
        "model": task.name(split, seed).split("/")[0], "split": split, "seed": seed,
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "per_class_recall": {e: float(np.mean(pred[yt == i] == i)) for i, e in enumerate(EMOTIONS)},
        "confusion": confusion_matrix(yt, pred, labels=range(len(EMOTIONS))).tolist(),
    }
    if "gate" in embs:
        res["mean_speech_gate"] = float(embs["gate"].mean())
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "test_outputs.npz", idx=te, y=yt, pred=pred,
             prob=torch.softmax(torch.tensor(logits), 1).numpy(), **{f"emb_{k}": v for k, v in embs.items()})
    print(f"{task.name(split, seed)}: test acc {res['accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}", flush=True)
    return res


def cli(modality, mode):
    import argparse
    from models.common.data import SPLITS
    ap = argparse.ArgumentParser(description=f"{mode} the {modality} pipeline")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--speech_feat", default="wavlm", choices=["wavlm", "mel"])
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    task = Task(modality, a.speech_feat, a.fusion)
    for split in a.splits:
        for seed in a.seeds:
            if mode == "train":
                train(task, split, seed, epochs=a.epochs)
            else:
                test(task, split, seed)
