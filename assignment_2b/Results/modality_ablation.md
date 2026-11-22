# Modality ablation (fusion models)

Test accuracy (%) when each test clip's text or speech is replaced by that of another random test clip (mean ± std over 3 seeds). Chance = 14.3 %.

| model | split | intact | text shuffled | speech shuffled |
|---|---|---|---|---|
| fusion_gated | OAF→YAF | 72.8 ± 3.7 | 72.8 ± 3.8 | 13.1 ± 0.5 |
| fusion_gated | YAF→OAF | 60.8 ± 1.3 | 60.7 ± 1.2 | 13.5 ± 0.4 |
| fusion_gated | Word-disjoint | 99.8 ± 0.1 | 99.8 ± 0.1 | 15.1 ± 0.7 |
| fusion_concat | OAF→YAF | 66.6 ± 5.4 | 66.6 ± 5.3 | 13.0 ± 0.4 |
| fusion_concat | YAF→OAF | 57.2 ± 1.3 | 57.2 ± 1.2 | 14.1 ± 0.5 |
| fusion_concat | Word-disjoint | 99.9 ± 0.1 | 99.9 ± 0.1 | 15.1 ± 0.7 |
