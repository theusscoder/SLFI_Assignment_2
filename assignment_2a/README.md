# Assignment 2a: Forced Alignment with Montreal Forced Aligner

A fully automated forced-alignment pipeline for the six provided utterances:
three news-reading clips from the BU Radio Corpus (speaker `f2b`, 25–31 s each) and
three non-native minimal-pair clips from the ISLE corpus ("I SAID WHITE NOT BAIT", …).

One command prepares the data, builds dictionaries (including G2P and hand-checked
pronunciations for missing words), runs six alignment experiments with two acoustic models
plus an adapted model, and writes the analysis tables and plots.

**Report:** [`report/Assignment_2a_Report.pdf`](report/Assignment_2a_Report.pdf)

## Results at a glance

| Experiment | Acoustic model | Text | Dictionary | Unknown (`spn`) phones |
|---|---|---|---|---|
| A | english_us_arpa | raw | stock | 6 |
| B | english_us_arpa | normalized | stock | 5 |
| **C** (main) | english_us_arpa | normalized | stock + G2P + manual | **0** |
| D | english_mfa | normalized | stock + G2P + manual | 0 |
| E | english_us_arpa **adapted** on this corpus | normalized | stock + G2P + manual | 0 |
| F | english_us_arpa (ISLE only) | normalized | learner pronunciation variants | 0 |

- All 6 utterances aligned in every experiment.
- ARPA vs english_mfa word boundaries: median difference 10 ms, 90 % within 20 ms. english_mfa
  stretches the last word of the short ISLE clips over the trailing silence (up to 2.4 s).
- Adaptation on ~1.5 min of audio changes boundaries by only 0.1 ms on average.
- Learner probe: given both options, the aligner picks **"bed" (`B EH1 D`) for "bad"**. LPC formants agree:
  the two vowels differ by only ~60 Hz in F1 (native /æ/–/ɛ/ differ by ~150–300 Hz).

Full tables are in [`results/`](results/); plots are in [`results/plots/`](results/plots/).

## Repository layout

```
assignment_2a/
├── data/
│   ├── wav/                 provided audio (BU files are NIST SPHERE despite the .wav name)
│   └── raw_transcripts/     provided transcripts, untouched
├── scripts/
│   ├── prepare_data.py      audio conversion + transcript normalization -> MFA corpus layout
│   ├── run_pipeline.py      the whole pipeline (calls the other two scripts)
│   └── analyze.py           statistics, model comparison, formants, plots
├── dictionaries/            OOV lists, G2P output, custom dictionaries, ISLE variant dictionary
├── outputs/<experiment>/    TextGrid files (one folder per experiment) + MFA logs
├── results/                 summary.md, boundary_agreement.md, suspicious_phones.md,
│                            isle_variants.md, isle_formants.md, plots/
└── report/                  PDF report
```

## 1. Install MFA

MFA is installed from conda-forge into its own environment (tested with MFA 3.4.3 on Windows 11).

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.11
conda activate mfa
mfa version
```

`matplotlib` and `numpy` (used by `analyze.py`) are already pulled in by MFA.

## 2. Prepare the data

```bash
python scripts/prepare_data.py
```

This writes `work/corpus_raw/` and `work/corpus_norm/` in MFA's `<speaker>/<utt>.wav + .lab` layout,
plus `work/data_report.json` with every change made. It handles two problems in the provided data:

1. **Audio format:** `F2BJRLP1-3.wav` are NIST SPHERE files (header `NIST_1A`, BU Radio Corpus), not RIFF WAV.
   They are converted to 16 kHz mono 16-bit PCM WAV by reading the SPHERE header directly.
2. **Transcripts:** line breaks and runs of spaces are collapsed; `seventy- six` → `seventy six`;
   `S.J.C.'s` → `s j c's` and `WBUR's` → `w b u r's` (read letter by letter); `de-politicize` → `depoliticize`;
   punctuation removed and text lower-cased.

## 3. Run the full pipeline

```bash
conda activate mfa
python scripts/run_pipeline.py              # ~15 min on a laptop CPU
python scripts/run_pipeline.py --skip-adapt # skip model adaptation
```

What it runs, in order (all commands are logged to `outputs/logs/`):

```bash
# pretrained models
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p        english_us_arpa
mfa model download acoustic   english_mfa
mfa model download dictionary english_us_mfa
mfa model download g2p        english_us_mfa

# pronunciations for out-of-vocabulary words
mfa g2p dictionaries/oov_arpa.txt english_us_arpa dictionaries/g2p_arpa.dict --num_pronunciations 1

# alignment (one call per experiment, e.g. the main experiment C)
mfa align work/corpus_norm dictionaries/custom_arpa.dict english_us_arpa outputs/C_arpa_normtext_g2pdict \
    --clean --beam 100 --retry_beam 400 --output_format long_textgrid

# adapt the acoustic model to this corpus, then align with it
mfa adapt work/corpus_norm dictionaries/custom_arpa.dict english_us_arpa work/english_us_arpa_adapted.zip
```

To re-run only the analysis: `python scripts/analyze.py`.

## 4. Inspect the TextGrids in Praat

Open the audio and its TextGrid together: in Praat, *Open → Read from file…* for
`work/corpus_norm/<speaker>/<utt>.wav` and `outputs/C_arpa_normtext_g2pdict/<speaker>/<utt>.TextGrid`,
select both, then *View & Edit*. The `words` and `phones` tiers appear under the spectrogram.

## Dictionaries

| File | Contents |
|---|---|
| `oov_arpa.txt`, `oov_mfa.txt` | words missing from each stock dictionary (raw + normalized text) |
| `g2p_*.dict` | pronunciations generated by the matching G2P model |
| `custom_*.dict` | stock dictionary + G2P entries + hand-checked overrides (`MANUAL_OVERRIDES` in `run_pipeline.py`) |
| `isle_learner_variants.dict` | alternative pronunciations for the ISLE minimal-pair words |
| `oov_summary.json` | everything above in one file, including what G2P produced before correction |
