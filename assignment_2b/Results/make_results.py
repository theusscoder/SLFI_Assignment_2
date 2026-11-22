"""Aggregate runs/*/metrics.json into accuracy tables and plots.

Run from assignment_2b/:  python Results/make_results.py
Writes Results/accuracy_tables.md, Results/all_runs.csv, Results/plots/*.png,
Results/failure_cases.md
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.common.data import EMOTIONS  # noqa: E402

RUNS, RES = ROOT / "runs", ROOT / "Results"
PLOTS = RES / "plots"
SPLIT_NAMES = {"random": "Random 80/10/10", "speaker_oaf2yaf": "Speaker: OAF→YAF",
               "speaker_yaf2oaf": "Speaker: YAF→OAF", "word_disjoint": "Word-disjoint"}
COLORS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#17becf", "#1f77b4"]


def load_runs():
    rows = []
    for f in RUNS.rglob("metrics.json"):
        rows.append({**json.loads(f.read_text()), "dir": str(f.parent)})
    return pd.DataFrame(rows)


def accuracy_tables(df):
    lines = ["# Test accuracy (mean ± std over seeds)\n"]
    for metric in ("accuracy", "macro_f1"):
        g = df.groupby(["model", "split"])[metric].agg(["mean", "std", "count"]).reset_index()
        pt = g.pivot(index="model", columns="split", values="mean")
        ps = g.pivot(index="model", columns="split", values="std")
        cols = [c for c in SPLIT_NAMES if c in pt.columns]
        lines.append(f"\n## {metric}\n\n| model | " + " | ".join(SPLIT_NAMES[c] for c in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for m in pt.index:
            lines.append(f"| {m} | " + " | ".join(
                f"{100 * pt.loc[m, c]:.1f} ± {100 * np.nan_to_num(ps.loc[m, c]):.1f}" for c in cols) + " |")
    if "mean_speech_gate" in df:
        g = df.dropna(subset=["mean_speech_gate"]).groupby(["model", "split"]).mean_speech_gate.mean()
        lines.append("\n## Gated fusion: mean gate value (1 = speech only, 0 = text only)\n")
        lines += [f"- {m} / {SPLIT_NAMES[s]}: {v:.3f}" for (m, s), v in g.items()]
    lines.append("\n## Per-class recall (random split, mean over seeds)\n")
    rec = df[df.split == "random"].groupby("model").per_class_recall.apply(
        lambda s: pd.DataFrame(list(s)).mean())
    rec = rec.unstack() if isinstance(rec.index, pd.MultiIndex) else rec
    lines.append("| model | " + " | ".join(EMOTIONS) + " |")
    lines.append("|---|" + "---|" * len(EMOTIONS))
    for m, r in rec.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{100 * r[e]:.1f}" for e in EMOTIONS) + " |")
    (RES / "accuracy_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def bar_plot(df):
    g = df.groupby(["model", "split"]).accuracy.agg(["mean", "std"]).reset_index()
    models = sorted(g.model.unique())
    splits = [s for s in SPLIT_NAMES if s in g.split.values]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = g[g.model == m].set_index("split").reindex(splits)
        ax.bar(np.arange(len(splits)) + i * w, 100 * sub["mean"], w, yerr=100 * sub["std"].fillna(0), label=m,
               capsize=2)
    ax.axhline(100 / 7, ls="--", c="k", lw=1)
    ax.text(len(splits) - 0.45, 100 / 7 + 1.5, "chance (14.3%)", fontsize=8)
    ax.set_xticks(np.arange(len(splits)) + 0.4 - w / 2)
    ax.set_xticklabels([SPLIT_NAMES[s] for s in splits])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Speech-only vs text-only vs multimodal, by evaluation protocol")
    fig.tight_layout()
    fig.savefig(PLOTS / "accuracy_by_split.png", dpi=140)
    plt.close(fig)


def confusion_plots(df):
    for (m, s), g in df.groupby(["model", "split"]):
        cm = np.sum([np.array(c) for c in g.confusion], axis=0).astype(float)
        cm /= cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        short = ["ang", "dis", "fear", "hap", "neu", "ps", "sad"]
        ax.set_xticks(range(7)); ax.set_xticklabels(short); ax.set_yticks(range(7)); ax.set_yticklabels(short)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{m}\n{SPLIT_NAMES[s]} (row %)", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / f"confusion_{m}_{s}.png", dpi=130)
        plt.close(fig)


def tsne_plots(df, split="speaker_oaf2yaf"):
    """Separability of the learned representation of each block, on the test set of one split."""
    panels = []
    for m in sorted(df.model.unique()):
        d = Path(df[(df.model == m) & (df.split == split) & (df.seed == 0)].dir.iloc[0])
        z = np.load(d / "test_outputs.npz")
        keys = [k for k in ("emb_speech", "emb_text", "emb_fusion", "emb_encoder") if k in z]
        for k in keys:
            label = {"emb_encoder": "speech: temporal block" if m.startswith("speech") else "text: contextual block",
                     "emb_speech": "fusion model: speech temporal block",
                     "emb_text": "fusion model: text contextual block",
                     "emb_fusion": "fusion model: fusion block"}[k]
            panels.append((f"{m}\n{label}", z[k], z["y"]))
    sil = []
    fig, axes = plt.subplots(2, (len(panels) + 1) // 2, figsize=(4.2 * ((len(panels) + 1) // 2), 8.4))
    for ax, (title, emb, y) in zip(axes.flat, panels):
        e2 = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
        s = silhouette_score(emb, y) if len(set(y)) > 1 else float("nan")
        sil.append((title.replace("\n", " | "), round(float(s), 3)))
        for i, emo in enumerate(EMOTIONS):
            ax.scatter(*e2[y == i].T, s=5, c=COLORS[i], label=emo)
        ax.set_title(f"{title}\nsilhouette = {s:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, markerscale=3, loc="best")
    fig.suptitle(f"t-SNE of learned representations, test set of split '{SPLIT_NAMES[split]}'", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / f"tsne_{split}.png", dpi=130)
    plt.close(fig)
    with open(RES / "accuracy_tables.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Silhouette score of block representations ({SPLIT_NAMES[split]} test set)\n\n")
        f.writelines(f"- {t}: {s}\n" for t, s in sil)


def failure_cases(df, model_prefix="fusion_gated", split="speaker_oaf2yaf", k=5):
    meta = pd.read_csv(ROOT / "cache" / "metadata.csv")
    d = Path(df[df.model.str.startswith(model_prefix) & (df.split == split) & (df.seed == 0)].dir.iloc[0])
    z = np.load(d / "test_outputs.npz")
    wrong = np.where(z["pred"] != z["y"])[0]
    conf = z["prob"][wrong, z["pred"][wrong]]
    lines = [f"# Failure cases: {d.relative_to(RUNS)} ({len(wrong)} errors / {len(z['y'])})\n",
             "Most confident mistakes first.\n",
             "| file | speaker | word | true | predicted | p(pred) | p(true) |", "|---|---|---|---|---|---|---|"]
    for j in wrong[np.argsort(-conf)][:k]:
        r = meta.iloc[z["idx"][j]]
        lines.append(f"| {Path(r.path).name} | {r.speaker} | {r.word} | {EMOTIONS[z['y'][j]]} | "
                     f"{EMOTIONS[z['pred'][j]]} | {z['prob'][j, z['pred'][j]]:.2f} | {z['prob'][j, z['y'][j]]:.2f} |")
    pairs = pd.Series([f"{EMOTIONS[a]} → {EMOTIONS[b]}" for a, b in zip(z["y"][wrong], z["pred"][wrong])])
    lines.append("\nMost frequent confusions:\n")
    lines += [f"- {p}: {n}" for p, n in pairs.value_counts().head(6).items()]
    (RES / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    df.drop(columns=["confusion", "per_class_recall"]).to_csv(RES / "all_runs.csv", index=False)
    accuracy_tables(df)
    bar_plot(df)
    confusion_plots(df)
    tsne_plots(df)
    failure_cases(df)


if __name__ == "__main__":
    main()
