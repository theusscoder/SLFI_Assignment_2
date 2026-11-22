# Failure cases: fusion_gated_wavlm\speaker_oaf2yaf\seed0 (335 errors / 1400)

Most confident mistakes first.

| file | speaker | word | true | predicted | p(pred) | p(true) |
|---|---|---|---|---|---|---|
| YAF_raid_happy.wav | YAF | raid | happy | angry | 0.98 | 0.01 |
| YAF_pad_happy.wav | YAF | pad | happy | angry | 0.97 | 0.01 |
| YAF_dab_happy.wav | YAF | dab | happy | angry | 0.97 | 0.01 |
| YAF_thought_angry.wav | YAF | thought | angry | disgust | 0.96 | 0.00 |
| YAF_raise_happy.wav | YAF | raise | happy | angry | 0.96 | 0.01 |

Most frequent confusions:

- happy → angry: 77
- angry → disgust: 59
- happy → fear: 46
- angry → sad: 33
- sad → fear: 30
- pleasant_surprise → happy: 23
