# SLFI Speech Processing Lab (LTRC, IIIT Hyderabad): Assignment 2

Submission by Shridhar Udayagiri.

| Part | Task | Folder | Report |
|---|---|---|---|
| **2a** | Forced alignment with Montreal Forced Aligner | [`assignment_2a/`](assignment_2a/) | [`Assignment_2a_Report.pdf`](assignment_2a/report/Assignment_2a_Report.pdf) |
| **2b** | Multimodal (speech + text) emotion recognition on TESS | [`assignment_2b/`](assignment_2b/) | [`Assignment_2b_Report.pdf`](assignment_2b/report/Assignment_2b_Report.pdf) |

Each folder has its own README with setup instructions, exact commands and results.

## Highlights

**2a**
- One-command MFA pipeline: data prep → OOV detection → G2P + hand-checked dictionary → 6 alignment experiments → analysis and plots.
- Found and fixed that the BU Radio Corpus files are NIST SPHERE audio with a `.wav` extension.
- Two acoustic models (`english_us_arpa`, `english_mfa`) plus a corpus-adapted model. Boundary agreement is measured, not just eyeballed.
- A learner-pronunciation probe shows the ISLE speaker says *bad* like *bed*, confirmed with LPC formant measurements.

**2b**
- Speech (WavLM + BiGRU + attention), text (BERT + Transformer) and gated multimodal fusion, in the required `models/*_pipeline/{train,test}.py` layout.
- Evaluated under four protocols (random, two speaker-independent directions, word-disjoint) × 3 seeds, because a random split on TESS overstates real performance.
- Shows that TESS transcripts ("Say the word X") carry no emotion information: text-only is at chance, and the fusion gate learns to rely on speech.
