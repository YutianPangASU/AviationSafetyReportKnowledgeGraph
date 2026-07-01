# Human vs LLM agreement — crane cohort labeling

Binary view (IN+MAYBE vs OUT). Positive class = IN (report/sentence belongs to the cohort).

### Document level — one label per (report, cohort), full dataset

| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|--:|
| Human IAA | 115 | 0.991 | 0.796 | – | – | – |
| Qwen3.6-35B | 1193 | 0.959 | 0.747 | 0.796 | 0.745 | 0.770 |
| Opus-4.8 | 1193 | 0.946 | 0.695 | 0.689 | 0.764 | 0.724 |
| Fable-5 | 1193 | 0.956 | 0.748 | 0.732 | 0.818 | 0.773 |

### Sentence level — one label per sentence, 300-report eval subset

| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|--:|
| Human IAA | 653 | 0.992 | 0.733 | – | – | – |
| Qwen3.6-35B | 1809 | 0.725 | 0.376 | 0.974 | 0.348 | 0.512 |
| Opus-4.8 | 1809 | 0.648 | 0.177 | 0.946 | 0.162 | 0.277 |
| Fable-5 | 1809 | 0.738 | 0.410 | 0.951 | 0.389 | 0.552 |

### Document level, per cohort (agreement / κ)

| Cohort | n | Qwen3.6-35B | Opus-4.8 | Fable-5 |
|---|--:|--:|--:|--:|
| Caused by Wind (IN=16) | 871 | 0.994 / 0.84 | 0.993 / 0.81 | 0.993 / 0.81 |
| Mobile Crane Accident (IN=57) | 170 | 0.859 / 0.68 | 0.800 / 0.58 | 0.853 / 0.68 |
| Static Crane Accident (IN=37) | 152 | 0.868 / 0.63 | 0.842 / 0.56 | 0.855 / 0.63 |
