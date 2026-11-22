"""End-to-end forced-alignment pipeline for Assignment 2a.

Run from the `mfa` conda environment:
    python scripts/run_pipeline.py            # everything
    python scripts/run_pipeline.py --skip-adapt

Steps
  1. prepare_data.py       -> work/corpus_raw, work/corpus_norm
  2. download pretrained acoustic models, dictionaries and G2P models
  3. find out-of-vocabulary (OOV) words for each dictionary
  4. generate pronunciations for OOVs with G2P -> custom dictionaries
  5. align every experiment configuration (see EXPERIMENTS)
  6. adapt the ARPA acoustic model to this corpus and align with it
  7. ISLE mispronunciation probe: dictionary with learner variants
Outputs go to outputs/<experiment>/<speaker>/<utt>.TextGrid
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
DICTS = ROOT / "dictionaries"

# name: (acoustic model, base dictionary, g2p model)
MODEL_SETS = {
    "arpa": ("english_us_arpa", "english_us_arpa", "english_us_arpa"),
    "mfa": ("english_mfa", "english_us_mfa", "english_us_mfa"),
}

# name: (corpus, model set, use g2p-augmented dictionary)
EXPERIMENTS = {
    "A_arpa_rawtext_basedict": ("corpus_raw", "arpa", False),
    "B_arpa_normtext_basedict": ("corpus_norm", "arpa", False),
    "C_arpa_normtext_g2pdict": ("corpus_norm", "arpa", True),
    "D_mfa_normtext_g2pdict": ("corpus_norm", "mfa", True),
}

# Hand-checked pronunciations that replace wrong G2P output (see
# dictionaries/oov_summary.json for what G2P produced on its own).
MANUAL_OVERRIDES = {
    "arpa": {
        "depoliticize": "D IY0 P AH0 L IH1 T AH0 S AY2 Z",
        "dukakis": "D UW0 K AA1 K AH0 S",
    },
    "mfa": {
        "c's": "s iː z",
        "r's": "ɑ ɹ z",
        "depoliticize": "d i p ə l ɪ t ə s aj z",
        "dukakis": "d u k ɑ k ə s",
    },
}

# Learner-style alternatives for the ISLE minimal pairs (ARPAbet). The
# aligner picks whichever pronunciation fits the audio best, so the chosen
# variant tells us what the speaker actually said.
ISLE_VARIANTS = {
    "white": ["W AY1 T", "V AY1 T", "HH W AY1 T"],
    "bait": ["B EY1 T", "B EH1 T"],
    "new": ["N UW1", "N Y UW1", "N OW1"],
    "no": ["N OW1", "N AO1", "N UW1"],
    "bad": ["B AE1 D", "B EH1 D", "B AE1 T"],
    "bed": ["B EH1 D", "B AE1 D", "B EH1 T"],
}


def run(cmd, log=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-4000:], sep="\n")
        raise SystemExit(f"command failed ({res.returncode}): {cmd[:3]}")
    return res.stdout


def pretrained_path(kind: str, name: str) -> Path:
    from montreal_forced_aligner.models import MODEL_TYPES
    return Path(MODEL_TYPES[kind].get_pretrained_path(name))


def download_models():
    for acoustic, dictionary, g2p in MODEL_SETS.values():
        run(["mfa", "model", "download", "acoustic", acoustic])
        run(["mfa", "model", "download", "dictionary", dictionary])
        run(["mfa", "model", "download", "g2p", g2p])


def read_dict_words(path: Path):
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            words.add(line.split("\t")[0].split()[0].lower())
    return words


def corpus_words(corpus: Path):
    words = []
    for lab in sorted(corpus.rglob("*.lab")):
        words += lab.read_text(encoding="utf-8").lower().split()
    return words


def clean_token(tok: str) -> str:
    # Mirror MFA's own punctuation stripping closely enough for OOV detection.
    return re.sub(r"^[^\w']+|[^\w']+$", "", tok.lower())


def build_dictionaries():
    """Find OOVs per model set and append G2P pronunciations to the base dictionary."""
    DICTS.mkdir(exist_ok=True)
    summary = {}
    for set_name, (_, dict_name, g2p_name) in MODEL_SETS.items():
        base = pretrained_path("dictionary", dict_name)
        vocab = read_dict_words(base)
        per_corpus = {}
        for corpus in ("corpus_raw", "corpus_norm"):
            toks = [clean_token(t) for t in corpus_words(WORK / corpus)]
            per_corpus[corpus] = sorted({t for t in toks if t and t not in vocab})
        oovs = sorted(set(per_corpus["corpus_raw"]) | set(per_corpus["corpus_norm"]))
        oov_file = DICTS / f"oov_{set_name}.txt"
        oov_file.write_text("\n".join(oovs) + "\n", encoding="utf-8")
        g2p_out = DICTS / f"g2p_{set_name}.dict"
        if oovs:
            run(["mfa", "g2p", oov_file, g2p_name, g2p_out, "--num_pronunciations", "1", "--clean"],
                log=OUT / "logs" / f"g2p_{set_name}.log")
        else:
            g2p_out.write_text("", encoding="utf-8")
        g2p_entries = dict(
            line.split("\t", 1) for line in g2p_out.read_text(encoding="utf-8").splitlines() if "\t" in line)
        overrides = MANUAL_OVERRIDES[set_name]
        added = {**{w: p for w, p in g2p_entries.items() if w not in overrides}, **overrides}
        custom = DICTS / f"custom_{set_name}.dict"
        with custom.open("w", encoding="utf-8") as f:
            f.write(base.read_text(encoding="utf-8").rstrip("\n") + "\n")
            f.writelines(f"{w}\t{p}\n" for w, p in added.items())
        summary[set_name] = {
            "base_dictionary": dict_name, "base_size": len(vocab),
            "oov_raw_text": per_corpus["corpus_raw"], "oov_normalized_text": per_corpus["corpus_norm"],
            "g2p_pronunciations": g2p_entries,
            "manual_overrides": overrides,
        }
    (DICTS / "oov_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def dictionary_for(set_name: str, use_g2p: bool) -> Path:
    if use_g2p:
        return DICTS / f"custom_{set_name}.dict"
    return pretrained_path("dictionary", MODEL_SETS[set_name][1])


def align(name, corpus: Path, dictionary, acoustic):
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    run(["mfa", "align", corpus, dictionary, acoustic, out,
         "--clean", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid",
         "--no_use_mp", "--num_jobs", "1"],
        log=OUT / "logs" / f"align_{name}.log")


def adapt_and_align():
    model = WORK / "english_us_arpa_adapted.zip"
    run(["mfa", "adapt", WORK / "corpus_norm", DICTS / "custom_arpa.dict", "english_us_arpa", model,
         "--clean", "--beam", "100", "--retry_beam", "400", "--no_use_mp", "--num_jobs", "1"],
        log=OUT / "logs" / "adapt_arpa.log")
    align("E_arpa_adapted_g2pdict", WORK / "corpus_norm", DICTS / "custom_arpa.dict", model)


def train_from_scratch():
    """Train an acoustic model on only these 6 files (~97 s) and align with it.

    Shows how much the pretrained models' thousands of hours of speech matter.
    """
    model = WORK / "scratch_arpa.zip"
    run(["mfa", "train", WORK / "corpus_norm", DICTS / "custom_arpa.dict", model,
         "--clean", "--no_use_mp", "--num_jobs", "1", "--output_directory", OUT / "G_scratch_trained_g2pdict",
         "--output_format", "long_textgrid"],
        log=OUT / "logs" / "train_scratch.log")


def isle_probe():
    corpus = WORK / "corpus_isle"
    if corpus.exists():
        shutil.rmtree(corpus)
    shutil.copytree(WORK / "corpus_norm" / "isle_sess0131", corpus / "isle_sess0131")
    variant_dict = DICTS / "isle_learner_variants.dict"
    lines = [ln for ln in (DICTS / "custom_arpa.dict").read_text(encoding="utf-8").splitlines()
             if ln.split("\t")[0].split()[0].lower() not in ISLE_VARIANTS]
    for word, prons in ISLE_VARIANTS.items():
        lines += [f"{word}\t{p}" for p in prons]
    variant_dict.write_text("\n".join(lines) + "\n", encoding="utf-8")
    align("F_isle_learner_variants", corpus, variant_dict, "english_us_arpa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-adapt", action="store_true")
    args = ap.parse_args()

    run([sys.executable, ROOT / "scripts" / "prepare_data.py"])
    download_models()
    summary = build_dictionaries()
    for set_name, s in summary.items():
        print(f"[{set_name}] OOV raw={s['oov_raw_text']}  normalized={s['oov_normalized_text']}")
    for name, (corpus, set_name, use_g2p) in EXPERIMENTS.items():
        align(name, WORK / corpus, dictionary_for(set_name, use_g2p), MODEL_SETS[set_name][0])
    if not args.skip_adapt:
        adapt_and_align()
    train_from_scratch()
    isle_probe()
    run([sys.executable, ROOT / "scripts" / "analyze.py"])


if __name__ == "__main__":
    main()

