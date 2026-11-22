# Test accuracy (mean ± std over seeds)


## accuracy

| model | Random 80/10/10 | Speaker: OAF→YAF | Speaker: YAF→OAF | Word-disjoint |
|---|---|---|---|---|
| fusion_concat_wavlm | 100.0 ± 0.0 | 66.6 ± 6.6 | 57.2 ± 1.6 | 99.9 ± 0.1 |
| fusion_gated_wavlm | 99.4 ± 0.5 | 72.8 ± 4.6 | 60.8 ± 1.6 | 99.8 ± 0.2 |
| speech_mel | 99.8 ± 0.2 | 47.2 ± 1.5 | 33.1 ± 3.7 | 99.8 ± 0.2 |
| speech_wavlm | 99.9 ± 0.2 | 70.5 ± 4.4 | 57.1 ± 6.5 | 99.5 ± 0.6 |
| text_bert | 13.3 ± 2.0 | 14.3 ± 0.0 | 14.3 ± 0.0 | 14.3 ± 0.0 |

## macro_f1

| model | Random 80/10/10 | Speaker: OAF→YAF | Speaker: YAF→OAF | Word-disjoint |
|---|---|---|---|---|
| fusion_concat_wavlm | 100.0 ± 0.0 | 62.6 ± 7.8 | 52.4 ± 0.6 | 99.9 ± 0.1 |
| fusion_gated_wavlm | 99.4 ± 0.5 | 69.5 ± 4.6 | 56.0 ± 1.3 | 99.8 ± 0.2 |
| speech_mel | 99.8 ± 0.2 | 43.9 ± 3.0 | 26.4 ± 6.2 | 99.8 ± 0.2 |
| speech_wavlm | 99.9 ± 0.2 | 66.4 ± 4.3 | 52.2 ± 6.3 | 99.5 ± 0.7 |
| text_bert | 5.7 ± 2.4 | 7.7 ± 1.8 | 5.6 ± 1.8 | 7.2 ± 1.2 |

## Gated fusion: mean gate value (1 = speech only, 0 = text only)

- fusion_gated_wavlm / Random 80/10/10: 0.612
- fusion_gated_wavlm / Speaker: OAF→YAF: 0.625
- fusion_gated_wavlm / Speaker: YAF→OAF: 0.610
- fusion_gated_wavlm / Word-disjoint: 0.619

## Per-class recall (random split, mean over seeds)

| model | angry | disgust | fear | happy | neutral | pleasant_surprise | sad |
|---|---|---|---|---|---|---|---|
| fusion_concat_wavlm | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fusion_gated_wavlm | 99.2 | 100.0 | 100.0 | 99.2 | 98.3 | 100.0 | 99.2 |
| speech_mel | 99.2 | 100.0 | 100.0 | 100.0 | 100.0 | 99.2 | 100.0 |
| speech_wavlm | 100.0 | 100.0 | 100.0 | 99.2 | 100.0 | 100.0 | 100.0 |
| text_bert | 0.0 | 26.7 | 5.8 | 23.3 | 1.7 | 33.3 | 2.5 |

## Silhouette score of block representations (Speaker: OAF→YAF test set)

- fusion_concat_wavlm | fusion model: speech temporal block: 0.41
- fusion_concat_wavlm | fusion model: text contextual block: -0.005
- fusion_concat_wavlm | fusion model: fusion block: 0.456
- fusion_gated_wavlm | fusion model: speech temporal block: 0.383
- fusion_gated_wavlm | fusion model: text contextual block: -0.005
- fusion_gated_wavlm | fusion model: fusion block: 0.432
- speech_mel | speech: temporal block: 0.323
- speech_wavlm | speech: temporal block: 0.369
- text_bert | text: contextual block: -0.005
