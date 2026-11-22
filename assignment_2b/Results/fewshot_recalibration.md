# Few-shot speaker recalibration

Encoder frozen; a logistic-regression classifier is refit on k labelled clips per emotion of the unseen speaker and tested on that speaker's remaining clips (3 seeds × 10 random draws). k = 0 is the original classifier trained on the other speaker.

| model | split | k=0 | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|---|
| speech_wavlm | OAF→YAF | 70.5 ± 3.6 | 93.1 ± 4.2 | 97.2 ± 1.3 | 99.1 ± 0.6 | 99.6 ± 0.2 | 99.8 ± 0.1 |
| speech_wavlm | YAF→OAF | 57.1 ± 5.3 | 75.7 ± 6.4 | 86.5 ± 4.0 | 92.9 ± 2.0 | 95.3 ± 1.0 | 96.9 ± 0.7 |
| fusion_gated_wavlm | OAF→YAF | 72.8 ± 3.7 | 90.2 ± 3.5 | 94.8 ± 2.4 | 97.8 ± 1.0 | 98.6 ± 0.4 | 99.2 ± 0.3 |
| fusion_gated_wavlm | YAF→OAF | 60.8 ± 1.3 | 76.5 ± 7.0 | 87.3 ± 3.2 | 92.2 ± 1.5 | 94.8 ± 0.8 | 96.2 ± 0.8 |
