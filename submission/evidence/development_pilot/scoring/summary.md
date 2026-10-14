# Saved-response scoring

Offline development analysis; no models executed.

Cases: 16; scenarios: 15; responses: 32.
Manual review: pending. Matching generation settings: True.
Response origins: {'saved_model': 32}.
Synthetic fixtures, when present, verify the workflow and are not model-performance results.
Accuracy uses scored facts only; inspect applicable and pending counts alongside it.

| Model | Dimension | Correct | Scored | Applicable | Needs review | Status |
|---|---|---:|---:|---:|---:|---|
| fine_tuned | numerical_extraction | 0 | 0 | 56 | 56 | needs_review |
| fine_tuned | arithmetic | 0 | 0 | 9 | 9 | needs_review |
| original | numerical_extraction | 0 | 0 | 56 | 56 | needs_review |
| original | arithmetic | 0 | 0 | 9 | 9 | needs_review |

fine_tuned: mean repeated word-trigram fraction 0.022; length-cap responses 0. Repetition is a signal, not a pass/fail score.

original: mean repeated word-trigram fraction 0.024; length-cap responses 0. Repetition is a signal, not a pass/fail score.

## Individual cases

| Model | Case | Numerical correct/scored | Pending numerical | Arithmetic correct/scored | Review |
|---|---|---:|---:|---:|---|
| fine_tuned | pilot-association-01 | 0/0 | 3 | 0/0 | pending |
| fine_tuned | pilot-association-02 | 0/0 | 4 | N/A | pending |
| fine_tuned | pilot-boolean-01 | 0/0 | 5 | 0/0 | pending |
| fine_tuned | pilot-boolean-02 | 0/0 | 5 | 0/0 | pending |
| fine_tuned | pilot-conversion-01 | 0/0 | 3 | 0/0 | pending |
| fine_tuned | pilot-conversion-02 | 0/0 | 2 | 0/0 | pending |
| fine_tuned | pilot-fairness-01 | 0/0 | 4 | 0/0 | pending |
| fine_tuned | pilot-fairness-02 | 0/0 | 4 | N/A | pending |
| fine_tuned | pilot-percentage-01 | 0/0 | 3 | 0/0 | pending |
| fine_tuned | pilot-percentage-02 | 0/0 | 3 | 0/0 | pending |
| fine_tuned | pilot-recommendation-01 | 0/0 | 2 | N/A | pending |
| fine_tuned | pilot-recommendation-02 | 0/0 | 2 | N/A | pending |
| fine_tuned | pilot-region-01 | 0/0 | 4 | N/A | pending |
| fine_tuned | pilot-threshold-01 | 0/0 | 4 | N/A | pending |
| fine_tuned | pilot-threshold-02 | 0/0 | 3 | N/A | pending |
| fine_tuned | pilot-traffic-01 | 0/0 | 5 | 0/0 | pending |
| original | pilot-association-01 | 0/0 | 3 | 0/0 | pending |
| original | pilot-association-02 | 0/0 | 4 | N/A | pending |
| original | pilot-boolean-01 | 0/0 | 5 | 0/0 | pending |
| original | pilot-boolean-02 | 0/0 | 5 | 0/0 | pending |
| original | pilot-conversion-01 | 0/0 | 3 | 0/0 | pending |
| original | pilot-conversion-02 | 0/0 | 2 | 0/0 | pending |
| original | pilot-fairness-01 | 0/0 | 4 | 0/0 | pending |
| original | pilot-fairness-02 | 0/0 | 4 | N/A | pending |
| original | pilot-percentage-01 | 0/0 | 3 | 0/0 | pending |
| original | pilot-percentage-02 | 0/0 | 3 | 0/0 | pending |
| original | pilot-recommendation-01 | 0/0 | 2 | N/A | pending |
| original | pilot-recommendation-02 | 0/0 | 2 | N/A | pending |
| original | pilot-region-01 | 0/0 | 4 | N/A | pending |
| original | pilot-threshold-01 | 0/0 | 4 | N/A | pending |
| original | pilot-threshold-02 | 0/0 | 3 | N/A | pending |
| original | pilot-traffic-01 | 0/0 | 5 | 0/0 | pending |

## Limitations

- Fact-line attribution only; prose needs manual review.
- Arithmetic checks the stated result, not reasoning or formula use.
- Accuracy denominators exclude pending facts; coverage is always reported.
- Repetition signals and declared generation metadata do not establish quality or provenance.
- No overall quality score or independent-scenario performance claim is produced.
