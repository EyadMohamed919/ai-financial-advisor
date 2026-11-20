# Intent Classifier

Decides what a chat message is for, so the message can be sent to the right component (issue #4).

| Notebook | What it does |
|---|---|
| `01_intent_dataset.ipynb` | Looks at the data: intents, dialects, examples |
| `02_intent_classifier.ipynb` | TF-IDF + Logistic Regression, evaluation, errors, the future-tense rule, the confidence threshold. Saves `models/intent_tfidf.joblib` |
| `03_intent_routing.ipynb` | Loads the saved model and shows where each message would go |

- Data: `tests/data/intent_dataset.csv` (2,008 messages in Egyptian Arabic, Franco-Arabic and English).
- Test accuracy: 89.1%.
- The working chat app that uses this classifier is in `ai-orchestrator/nlp_prototype/`.
