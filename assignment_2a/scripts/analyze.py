"""Analyse the TextGrids produced by run_pipeline.py.

Writes to results/:
  summary.json / summary.md     per-experiment alignment statistics
  boundary_agreement.md         word-boundary differences between models
  suspicious_phones.md          phones at MFA's minimum duration or unusually long
  isle_variants.md              which learner pronunciation the aligner chose
  plots/*.png                   waveform + spectrogram + word/phone tiers
"""
import json
import re
import wave
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = ROOT / "results"
PLOTS = RES / "plots"
SILENCE = {"", "sil", "sp", "<eps>"}
UNKNOWN = {"spn", "<unk>"}
MIN_PHONE = 0.031  # 3 HMM states x 10 ms frames, plus float slack
LONG_PHONE = 0.30


def parse_textgrid(path: Path):
    """Minimal long-format TextGrid reader -> {tier_name: [(start, end, label), ...]}."""
    tiers, name, cur = {}, None, None
    xmin = xmax = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"')
            cur = tiers.setdefault(name, [])
        elif line.startswith("xmin =") and cur is not None:
            xmin = float(line.split("=")[1])
        elif line.startswith("xmax =") and cur is not None:
            xmax = float(line.split("=")[1])
        elif line.startswith("text =") and cur is not None:
            cur.append((xmin, xmax, line.split("=", 1)[1].strip().strip('"')))
    return tiers


def load_experiments():
    exps = {}
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "logs"):
        utts = {}
        for tg in d.rglob("*.TextGrid"):
            t = parse_textgrid(tg)
            utts[tg.stem] = {"words": t.get("words", []), "phones": t.get("phones", [])}
        exps[d.name] = utts
    return exps


def speech(intervals):
    return [iv for iv in intervals if iv[2].lower() not in SILENCE]


def read_wav(path: Path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return w.getframerate(), x


def all_utts():
    return sorted(p.stem for p in (ROOT / "work" / "corpus_norm").rglob("*.wav"))


def stats(exps):
    rows = {}
    utts = all_utts()
    for name, data in exps.items():
        words = [w for u in data.values() for w in speech(u["words"])]
        phones = [p for u in data.values() for p in speech(u["phones"])]
        durs = np.array([p[1] - p[0] for p in phones]) if phones else np.array([0.0])
        rows[name] = {
            "utterances_aligned": f"{len(data)}/{len(utts)}",
            "missing": sorted(set(utts) - set(data)),
            "words": len(words),
            "unknown_words": sum(w[2].lower() in UNKNOWN for w in words),
            "spn_phones": sum(p[2].lower() in UNKNOWN for p in phones),
            "phones": len(phones),
            "min_duration_phones_pct": round(100 * float(np.mean(durs <= MIN_PHONE)), 1),
            "long_phones": int(np.sum(durs > LONG_PHONE)),
            "mean_phone_ms": round(1000 * float(durs.mean()), 1),
        }
    return rows


def boundary_agreement(exps):
    """Mean |difference| of word start/end times for experiments with identical word sequences."""
    out = []
    for a, b in combinations(exps, 2):
        diffs = []
        for utt in set(exps[a]) & set(exps[b]):
            wa, wb = speech(exps[a][utt]["words"]), speech(exps[b][utt]["words"])
            if [w[2] for w in wa] != [w[2] for w in wb]:
                continue
            for x, y in zip(wa, wb):
                diffs += [abs(x[0] - y[0]), abs(x[1] - y[1])]
        if diffs:
            d = np.array(diffs) * 1000
            out.append((a, b, len(d), round(float(d.mean()), 1), round(float(np.median(d)), 1),
                        round(float(np.mean(d <= 20) * 100), 1), round(float(d.max()), 1)))
    return out


def suspicious_phones(exps, name):
    rows = []
    for utt, u in sorted(exps.get(name, {}).items()):
        words = speech(u["words"])
        for s, e, lab in speech(u["phones"]):
            d = e - s
            if d <= MIN_PHONE or d > LONG_PHONE:
                word = next((w[2] for w in words if w[0] <= s < w[1]), "?")
                rows.append((utt, word, lab, round(s, 3), round(d * 1000)))
    return rows


def word_phones(u, word):
    return [p[2] for p in speech(u["phones"]) if word[0] <= p[0] < word[1]]


def isle_choices(exps):
    rows = []
    for exp in ("C_arpa_normtext_g2pdict", "D_mfa_normtext_g2pdict", "F_isle_learner_variants"):
        for utt, u in sorted(exps.get(exp, {}).items()):
            if not utt.startswith("ISLE"):
                continue
            words = speech(u["words"])
            # The two contrasted words are the 3rd and 5th: "I SAID <x> NOT <y>".
            for w in words[2:3] + words[4:5]:
                rows.append((exp, utt[-11:], w[2], " ".join(word_phones(u, w)),
                             round(w[0], 2), round((w[1] - w[0]) * 1000)))
    return rows


def formants(x, sr, order=12, target_sr=11000):
    """F1/F2 (Hz) of a short frame via LPC root-finding, Praat-style.

    The frame is resampled to ~11 kHz first (Praat's default 5.5 kHz formant
    ceiling) so the 12 LPC poles model the 0-5.5 kHz region where formants live.
    """
    n = int(len(x) * target_sr / sr)
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hamming(n)  # pre-emphasis + window
    r = np.correlate(x, x, "full")[n - 1:n + order]
    a, e = np.array([1.0]), r[0]
    for i in range(1, order + 1):  # Levinson-Durbin
        k = -(r[i] + a[1:] @ r[i - 1:0:-1]) / e
        a = np.append(a, 0) + k * np.append(a, 0)[::-1]
        e *= 1 - k * k
    poles = sorted((np.angle(z) * target_sr / (2 * np.pi), -target_sr / np.pi * np.log(abs(z)))
                   for z in np.roots(a) if z.imag > 0)
    freqs = [round(f) for f, bw in poles if f > 150 and bw < 500]  # narrow-bandwidth poles only
    return (freqs[0], freqs[1]) if len(freqs) >= 2 else (None, None)


def isle_formants(exps, exp="C_arpa_normtext_g2pdict"):
    """F1/F2 from 25 ms frames at 30/50/70 % of each contrasted vowel (shows diphthong movement)."""
    rows = []
    for utt, u in sorted(exps.get(exp, {}).items()):
        if not utt.startswith("ISLE"):
            continue
        sr, x = read_wav(next((ROOT / "work" / "corpus_norm").rglob(f"{utt}.wav")))
        words = speech(u["words"])
        for w in words[2:3] + words[4:5]:
            for s, e, lab in speech(u["phones"]):
                if w[0] <= s < w[1] and lab[-1].isdigit():
                    pts = []
                    for frac in (0.3, 0.5, 0.7):
                        c = s + frac * (e - s)
                        f1, f2 = formants(x[int((c - 0.0125) * sr):int((c + 0.0125) * sr)], sr)
                        pts.append(f"{f1}/{f2}")
                    rows.append((utt[-11:], w[2], lab, *pts))
    return rows


def plot_alignment(utt, exp_names, exps, start=None, end=None, fname=None):
    wav = next((ROOT / "work" / "corpus_norm").rglob(f"{utt}.wav"))
    sr, x = read_wav(wav)
    t0 = start or 0.0
    t1 = end or len(x) / sr
    seg = x[int(t0 * sr):int(t1 * sr)]
    t = np.arange(len(seg)) / sr + t0
    tiers = [(n, kind) for n in exp_names for kind in ("words", "phones")]
    fig, axes = plt.subplots(2 + len(tiers), 1, figsize=(14, 3.2 + 0.55 * len(tiers)), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 2] + [0.55] * len(tiers)})
    axes[0].plot(t, seg, lw=0.4, color="#334")
    axes[0].set_ylabel("wave")
    axes[1].specgram(seg, NFFT=400, Fs=sr, noverlap=240, xextent=(t0, t1), cmap="magma", vmin=-120)
    axes[1].set_ylim(0, 8000)
    axes[1].set_ylabel("Hz")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for ax, (n, kind) in zip(axes[2:], tiers):
        c = colors[exp_names.index(n) % len(colors)]
        for s, e, lab in exps[n][utt][kind]:
            if e < t0 or s > t1:
                continue
            ax.axvline(s, color=c, lw=0.8)
            if lab.lower() not in SILENCE:
                ax.text((max(s, t0) + min(e, t1)) / 2, 0.5, lab, ha="center", va="center",
                        fontsize=8 if kind == "words" else 7)
                for a in axes[:2]:
                    a.axvline(s, color=c, lw=0.5, alpha=0.5)
        ax.set_yticks([])
        ax.set_ylabel(f"{n.split('_')[0]} {kind[0]}", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlim(t0, t1)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{utt}  [{t0:.1f}-{t1:.1f} s]  experiments: {', '.join(exp_names)}", fontsize=10)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / (fname or f"{utt}.png"), dpi=130)
    plt.close(fig)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    exps = load_experiments()
    RES.mkdir(exist_ok=True)
    s = stats(exps)
    (RES / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    keys = list(next(iter(s.values())).keys())
    (RES / "summary.md").write_text(md_table(["experiment"] + keys, [[k] + list(v.values()) for k, v in s.items()]),
                                    encoding="utf-8")

    ba = boundary_agreement(exps)
    (RES / "boundary_agreement.md").write_text(md_table(
        ["exp A", "exp B", "boundaries", "mean ms", "median ms", "% within 20 ms", "max ms"], ba), encoding="utf-8")

    sp = suspicious_phones(exps, "C_arpa_normtext_g2pdict")
    (RES / "suspicious_phones.md").write_text(
        f"Phones at MFA's minimum duration (<= {MIN_PHONE * 1000:.0f} ms) or longer than {LONG_PHONE * 1000:.0f} ms "
        "in experiment C.\n\n" + md_table(["utt", "word", "phone", "start s", "dur ms"], sp), encoding="utf-8")

    ic = isle_choices(exps)
    (RES / "isle_variants.md").write_text(md_table(["experiment", "utt", "word", "aligned phones", "start s", "dur ms"], ic),
                                          encoding="utf-8")

    fm = isle_formants(exps)
    (RES / "isle_formants.md").write_text(
        "F1/F2 (Hz) by LPC on 25 ms frames at 30/50/70 % of each contrasted vowel (experiment C boundaries).\n"
        "Typical native values: /ae/ (bad) has a clearly higher F1 than /eh/ (bed) (roughly 650-1000 vs "
        "500-650 Hz depending on speaker); /ay/ (white) and /ey/ (bait) are diphthongs, so F1/F2 should move.\n\n"
        + md_table(["utt", "word", "dictionary vowel", "F1/F2 @30%", "F1/F2 @50%", "F1/F2 @70%"], fm),
        encoding="utf-8")

    main_exp = "C_arpa_normtext_g2pdict"
    compare = [e for e in ("C_arpa_normtext_g2pdict", "D_mfa_normtext_g2pdict", "E_arpa_adapted_g2pdict") if e in exps]
    for utt in all_utts():
        if utt not in exps.get(main_exp, {}):
            continue
        if utt.startswith("ISLE"):
            plot_alignment(utt, [e for e in (main_exp, "F_isle_learner_variants") if utt in exps.get(e, {})], exps)
        else:
            plot_alignment(utt, [main_exp], exps, 0.0, 4.0, f"{utt}_first4s.png")
            plot_alignment(utt, [e for e in compare if utt in exps[e]], exps, 4.0, 7.0, f"{utt}_compare_models.png")
    print((RES / "summary.md").read_text(encoding="utf-8"))
    print((RES / "boundary_agreement.md").read_text(encoding="utf-8"))
    print((RES / "isle_variants.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
