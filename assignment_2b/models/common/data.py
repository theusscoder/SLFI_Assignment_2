"""TESS loading, preprocessing and evaluation splits shared by all three pipelines.

TESS: 2 speakers (OAF ~64 y, YAF ~26 y) x 200 target words x 7 emotions = 2800 clips.
Every clip is the carrier phrase "Say the word <word>", so the transcript is
built from the file name.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
SPLITS = ["random", "speaker_oaf2yaf", "speaker_yaf2oaf", "word_disjoint"]

_EMO_ALIASES = {"ps": "pleasant_surprise", "pleasant_surprised": "pleasant_surprise",
                "pleasant_surprise": "pleasant_surprise", "fear": "fear", "fearful": "fear",
                "disgust": "disgust", "disgusted": "disgust", "angry": "angry", "happy": "happy",
                "neutral": "neutral", "sad": "sad"}


def load_metadata(data_dir: Path = DATA) -> pd.DataFrame:
    rows, seen = [], set()
    for wav in sorted(data_dir.rglob("*.wav")):
        # "OA_bite_neutral.wav" in the official release is missing the F of OAF.
        m = re.match(r"^(OAF?|YAF)_(.+?)_([A-Za-z_]+)$", wav.stem, re.I)
        if not m:
            continue
        spk, word, emo = m.group(1).upper().replace("OA", "OAF").replace("OAFF", "OAF"), m.group(2).lower(), \
            m.group(3).lower()
        emo = _EMO_ALIASES.get(emo)
        # The Kaggle archive ships the corpus twice (two nested copies); keep one.
        if emo is None or (spk, word, emo) in seen:
            continue
        seen.add((spk, word, emo))
        rows.append({"path": str(wav), "speaker": spk, "word": word, "emotion": emo,
                     "label": EMOTIONS.index(emo), "text": f"Say the word {word}"})
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No TESS wav files found under {data_dir}")
    return df


def make_split(df: pd.DataFrame, split: str, seed: int = 0):
    """Return (train_idx, val_idx, test_idx) as numpy index arrays."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    if split == "random":
        train, val, test = [], [], []
        for _, g in df.groupby(["speaker", "emotion"]):  # stratified 80/10/10
            p = rng.permutation(g.index.values)
            n = len(p)
            test += list(p[:n // 10]); val += list(p[n // 10:n // 5]); train += list(p[n // 5:])
        return np.array(train), np.array(val), np.array(test)
    if split.startswith("speaker_"):
        src = "OAF" if split == "speaker_oaf2yaf" else "YAF"
        tr = idx[df.speaker.values == src]
        te = idx[df.speaker.values != src]
        p = rng.permutation(tr)
        return p[len(p) // 10:], p[:len(p) // 10], te
    if split == "word_disjoint":
        words = rng.permutation(sorted(df.word.unique()))
        n = len(words)
        test_w, val_w = set(words[:n // 5]), set(words[n // 5:n // 5 + n // 10])
        w = df.word.values
        return (idx[~np.isin(w, list(test_w | val_w))], idx[np.isin(w, list(val_w))],
                idx[np.isin(w, list(test_w))])
    raise ValueError(split)
