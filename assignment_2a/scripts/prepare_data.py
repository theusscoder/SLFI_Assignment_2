"""Prepare the provided 2a dataset in the layout MFA expects.

MFA wants   corpus/<speaker>/<utt>.wav  +  corpus/<speaker>/<utt>.lab

Two issues in the provided data are handled here:
  1. The BU Radio Corpus files (F2BJRLP*.wav) are NIST SPHERE files with a
     .wav extension, not RIFF/WAV. They are converted to 16 kHz mono PCM WAV.
  2. The raw transcripts contain line breaks, irregular spacing, split
     numbers ("seventy- six"), abbreviations ("S.J.C.'s", "WBUR's") and
     hyphenated words. They are normalized to how the words were spoken.

Two corpora are written so the effect of normalization can be compared:
  corpus_raw/   transcripts only whitespace-collapsed (MFA's own cleanup)
  corpus_norm/  transcripts normalized by the rules below
"""
import argparse
import json
import re
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Rewrites applied before generic cleanup, in order. Each entry is
# (pattern, replacement, reason) so the report can list every change.
RULES = [
    (r"\bS\.J\.C\.'s\b", "S J C's", "initialism read letter by letter"),
    (r"\bWBUR's\b", "W B U R's", "radio call sign read letter by letter"),
    (r"(\w)-\s+(\w)", r"\1 \2", "number split across a hyphen and space"),
    (r"\bde-politicize\b", "depoliticize", "prefix hyphen joined into one word"),
]


def speaker_of(utt: str) -> str:
    # BU Radio Corpus ids start with the speaker (f2b); ISLE ids carry the session.
    if utt.startswith("F2B"):
        return "bu_f2b"
    m = re.match(r"ISLE_(SESS\d+)", utt)
    return f"isle_{m.group(1).lower()}" if m else "unknown"


def read_audio(path: Path):
    """Return (sample_rate, channels, sample_width, pcm_bytes) for RIFF or NIST SPHERE."""
    data = path.read_bytes()
    if data[:4] == b"RIFF":
        with wave.open(str(path)) as w:
            return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes()), "riff"
    if data[:7] == b"NIST_1A":
        header_len = int(data[8:16].strip())
        header = data[:header_len].decode("ascii", errors="ignore")
        fields = {}
        for line in header.splitlines()[2:]:
            parts = line.split(None, 2)
            if len(parts) == 3:
                key, typ, val = parts
                fields[key] = int(val) if typ == "-i" else val
        if fields.get("sample_coding", "pcm") != "pcm":
            raise ValueError(f"{path.name}: unsupported SPHERE coding {fields['sample_coding']}")
        width = fields["sample_n_bytes"]
        pcm = data[header_len:header_len + fields["sample_count"] * width * fields["channel_count"]]
        if fields.get("sample_byte_format") == "10" and width == 2:  # big-endian -> little-endian
            pcm = b"".join(pcm[i + 1:i + 2] + pcm[i:i + 1] for i in range(0, len(pcm), 2))
        return fields["sample_rate"], fields["channel_count"], width, pcm, "nist_sphere"
    raise ValueError(f"{path.name}: unknown audio format {data[:8]!r}")


def write_wav(path: Path, rate, channels, width, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str):
    text = collapse(text)
    applied = []
    for pattern, repl, reason in RULES:
        new = re.sub(pattern, repl, text)
        if new != text:
            applied.append({"rule": pattern, "reason": reason})
        text = new
    # Drop punctuation except word-internal apostrophes and hyphens.
    text = re.sub(r"[^\w'\- ]", " ", text)
    text = re.sub(r"(?<!\w)['\-]|['\-](?!\w)", " ", text)
    return collapse(text).lower(), applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "work")
    args = ap.parse_args()

    report = []
    for wav in sorted((args.data / "wav").glob("*.wav")):
        utt = wav.stem
        txt = next((args.data / "raw_transcripts").glob(f"{utt}.*"))
        raw = txt.read_text(encoding="utf-8", errors="ignore")
        rate, ch, width, pcm, fmt = read_audio(wav)
        spk = speaker_of(utt)
        norm, applied = normalize(raw)
        for variant, text in (("corpus_raw", collapse(raw)), ("corpus_norm", norm)):
            d = args.out / variant / spk
            d.mkdir(parents=True, exist_ok=True)
            write_wav(d / f"{utt}.wav", rate, ch, width, pcm)
            (d / f"{utt}.lab").write_text(text + "\n", encoding="utf-8")
        report.append({
            "utt": utt, "speaker": spk, "source_format": fmt, "sample_rate": rate,
            "channels": ch, "bits": width * 8,
            "duration_s": round(len(pcm) / (rate * ch * width), 2),
            "raw": collapse(raw), "normalized": norm, "rules_applied": applied,
        })
        print(f"{utt:38s} {fmt:12s} {rate} Hz {len(pcm) / (rate * ch * width):6.2f} s  -> {spk}")

    (args.out / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
