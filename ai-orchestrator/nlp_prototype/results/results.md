Train: 1599 synthetic · Test: 401 synthetic (held out) + 80 hand-written

| Approach | Set | Type | Intent | Amount | Category | End-to-end |
|---|---|---|---|---|---|---|
| A. Rules (keywords + regex) | synthetic test | 78.3% | 76.8% | 100.0% | 44.7% | **46.4%** |
| | hand-written | 92.5% | – | 100.0% | 86.8% | **83.8%** |
| B. TF-IDF + Logistic Regression | synthetic test | 88.0% | 87.3% | 100.0% | 79.6% | **76.8%** |
| | hand-written | 100.0% | – | 100.0% | 92.1% | **92.5%** |
| C. spaCy (NER + textcat) | synthetic test | 88.8% | 88.0% | 99.7% | 71.1% | **72.8%** |
| | hand-written | 97.5% | – | 100.0% | 80.3% | **81.2%** |

| Approach | Model size | Latency / message | Training time |
|---|---|---|---|
| A. Rules (keywords + regex) | 0.0 MB | 0.04 ms | 0.0 s |
| B. TF-IDF + Logistic Regression | 7.23 MB | 0.88 ms | 4.5 s |
| C. spaCy (NER + textcat) | 34.72 MB | 3.31 ms | 146.7 s |

spaCy NER on synthetic test (exact span match): precision 84.2%, recall 79.6%, F1 **81.8%**. Entities skipped in training because of token misalignment: 98.
