"""Two follow-up experiments on the trained models (no retraining of the encoders).

1. Modality ablation: evaluate trained fusion models with the text (or the speech)
   of each test clip swapped with another random test clip's. If accuracy does not
   change when text is shuffled, the model is not using text.
2. Few-shot speaker recalibration: on the unseen-speaker splits, keep the encoder
   frozen and refit only a linear classifier on k labelled clips per emotion from
   the new speaker; evaluate on that speaker's remaining clips. Tests whether the
   cross-speaker errors are mislabelled-but-separable clusters.

Run from assignment_2b/:  python Results/extra_experiments.py
Writes Results/modality_ablation.md, Results/fewshot_recalibration.md, Results/plots/fewshot.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.common.data import make_split  # noqa: E402
from models.common.engine import DEV, RUNS, Task, pad  # noqa: E402

RES = ROOT / "Results"
SPLITS = ["speaker_oaf2yaf", "speaker_yaf2oaf", "word_disjoint"]
SPEAKER_SPLITS = ["speaker_oaf2yaf", "speaker_yaf2oaf"]
SEEDS = [0, 1, 2]
SPLIT_NAMES = {"speaker_oaf2yaf": "OAF→YAF", "speaker_yaf2oaf": "YAF→OAF", "word_disjoint": "Word-disjoint"}


@torch.no_grad()
def fusion_accuracy(task, model, te, speech_idx, text_idx, bs=64):
    model.eval()
    correct = 0
    for b in range(0, len(te), bs):
        s = [task.S[i] for i in speech_idx[b:b + bs]]
        t = [task.T[i] for i in text_idx[b:b + bs]]
        logits, _ = model(*pad(s), *pad(t))
        correct += (logits.argmax(1).cpu().numpy() == task.y[te[b:b + bs]]).sum()
    return correct / len(te)


def modality_ablation():
    lines = ["# Modality ablation (fusion models)\n",
             "Test accuracy (%) when each test clip's text or speech is replaced by that of another random test clip "
             "(mean ± std over 3 seeds). Chance = 14.3 %.\n",
             "| model | split | intact | text shuffled | speech shuffled |", "|---|---|---|---|---|"]
    for fusion in ("gated", "concat"):
        task = Task("fusion", "wavlm", fusion)
        for split in SPLITS:
            res = []
            for seed in SEEDS:
                _, _, te = make_split(task.meta, split, seed)
                model = task.build()
                model.load_state_dict(torch.load(RUNS / task.name(split, seed) / "model.pt", map_location=DEV))
                perm = np.random.default_rng(seed).permutation(te)
                res.append([fusion_accuracy(task, model, te, te, te),
                            fusion_accuracy(task, model, te, te, perm),
                            fusion_accuracy(task, model, te, perm, te)])
            m, s = 100 * np.mean(res, 0), 100 * np.std(res, 0)
            lines.append(f"| fusion_{fusion} | {SPLIT_NAMES[split]} | " +
                         " | ".join(f"{a:.1f} ± {b:.1f}" for a, b in zip(m, s)) + " |")
            print(lines[-1], flush=True)
    (RES / "modality_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fewshot():
    ks = [1, 2, 5, 10, 20]
    reps = 10
    models = {"speech_wavlm": "emb_encoder", "fusion_gated_wavlm": "emb_fusion"}
    table = {}
    for model, key in models.items():
        for split in SPEAKER_SPLITS:
            zero, curves = [], {k: [] for k in ks}
            for seed in SEEDS:
                z = np.load(RUNS / model / split / f"seed{seed}" / "test_outputs.npz")
                emb, y, pred = z[key], z["y"], z["pred"]
                zero.append(np.mean(pred == y))
                for k in ks:
                    for r in range(reps):
                        rng = np.random.default_rng(1000 * seed + r)
                        shots = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(7)])
                        rest = np.setdiff1d(np.arange(len(y)), shots)
                        sc = StandardScaler().fit(emb[shots])
                        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(emb[shots]), y[shots])
                        curves[k].append(clf.score(sc.transform(emb[rest]), y[rest]))
            table[(model, split)] = (np.mean(zero), np.std(zero),
                                     {k: (np.mean(v), np.std(v)) for k, v in curves.items()})
    lines = ["# Few-shot speaker recalibration\n",
             "Encoder frozen; a logistic-regression classifier is refit on k labelled clips per emotion of the "
             "unseen speaker and tested on that speaker's remaining clips (3 seeds × 10 random draws). "
             "k = 0 is the original classifier trained on the other speaker.\n",
             "| model | split | k=0 | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    for (model, split), (z0, zs, cur) in table.items():
        lines.append(f"| {model} | {SPLIT_NAMES[split]} | {100 * z0:.1f} ± {100 * zs:.1f} | " +
                     " | ".join(f"{100 * cur[k][0]:.1f} ± {100 * cur[k][1]:.1f}" for k in ks) + " |")
    (RES / "fewshot_recalibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4))
    for (model, split), (z0, _, cur) in table.items():
        xs = [0] + ks
        ys = [100 * z0] + [100 * cur[k][0] for k in ks]
        ax.plot(xs, ys, marker="o", ls="-" if model.startswith("speech") else "--",
                label=f"{model} · {SPLIT_NAMES[split]}")
    ax.axhline(100 / 7, c="k", lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0] + ks)
    ax.set_xticklabels(["0"] + [str(k) for k in ks])
    ax.set_xlabel("labelled clips per emotion from the new speaker (k)")
    ax.set_ylabel("accuracy on the new speaker (%)")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8)
    ax.set_title("Few-shot recalibration of the classifier (encoder frozen)")
    fig.tight_layout()
    fig.savefig(RES / "plots" / "fewshot.png", dpi=140)


if __name__ == "__main__":
    modality_ablation()
    fewshot()
