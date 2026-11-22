"""Preprocessing + frozen feature extraction, cached to disk once.

Speech (time_steps x features):
  - load, resample to 16 kHz mono, trim leading/trailing silence (top_db=30),
    peak-normalize, crop/pad to at most 3.0 s
  - "wavlm": hidden states of a frozen WavLM-Base+ layer (50 frames/s x 768)
  - "mel":   80-band log-mel spectrogram (100 frames/s x 80), as a lighter baseline
Text (tokens x features):
  - lower-case, BERT WordPiece tokenization
  - "bert": frozen bert-base-uncased last hidden layer (tokens x 768)

Run:  python -m models.common.features
"""
import argparse

import librosa
import numpy as np
import torch

from models.common.data import CACHE, load_metadata

SR = 16000
MAX_SEC = 3.0
WAVLM = "microsoft/wavlm-base-plus"
WAVLM_LAYER = 9  # mid-upper layers carry the most paralinguistic information in SUPERB-style probes
BERT = "bert-base-uncased"


def preprocess_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.abs(y).max() + 1e-8) * 0.95
    return y[: int(MAX_SEC * SR)]


def save_ragged(name, arrays):
    lengths = np.array([len(a) for a in arrays])
    np.savez(CACHE / f"{name}.npz", data=np.concatenate(arrays).astype(np.float16), lengths=lengths)


def load_ragged(name):
    z = np.load(CACHE / f"{name}.npz")
    data, lengths = z["data"], z["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(lengths))]


@torch.no_grad()
def extract(which):
    from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor

    CACHE.mkdir(exist_ok=True)
    df = load_metadata()
    df.to_csv(CACHE / "metadata.csv", index=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feats = []
    if which == "mel":
        for p in df.path:
            y = preprocess_audio(p)
            m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=80)
            feats.append(np.log(m + 1e-6).T)
    elif which == "wavlm":
        fe = AutoFeatureExtractor.from_pretrained(WAVLM)
        model = AutoModel.from_pretrained(WAVLM).to(dev).eval()
        for i, p in enumerate(df.path):
            x = fe(preprocess_audio(p), sampling_rate=SR, return_tensors="pt").input_values.to(dev)
            h = model(x, output_hidden_states=True).hidden_states[WAVLM_LAYER][0]
            feats.append(h.cpu().numpy())
            if i % 400 == 0:
                print(f"wavlm {i}/{len(df)}", flush=True)
    elif which == "bert":
        tok = AutoTokenizer.from_pretrained(BERT)
        model = AutoModel.from_pretrained(BERT).to(dev).eval()
        for t in df.text:
            enc = tok(t.lower(), return_tensors="pt").to(dev)
            feats.append(model(**enc).last_hidden_state[0].cpu().numpy())
    save_ragged(which, feats)
    print(which, len(feats), "items, example shape", feats[0].shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["bert", "mel", "wavlm"])
    for w in ap.parse_args().which:
        extract(w)
