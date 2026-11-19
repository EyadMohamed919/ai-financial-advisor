# NLP prototype: understanding Egyptian expense messages

This prototype turns a chat message like

> انا اشتريت كوتشي نايك من مول مصر ب 1200 جنيه

into a structured transaction and a reply:

```json
{"intent": "log_expense", "amount": 1200, "currency": "EGP", "category": "Shopping",
 "category_confidence": 0.949, "item": "كوتشي نايك", "merchant": "مول مصر", "missing": []}
```
> اتسجل ✅ مصروف: 1,200 جنيه – مشتريات (كوتشي نايك من مول مصر)

The goal is to find the best **lightweight, free, CPU-only** approach before building the real service. No LLM API is used.

## Approaches compared

| | How it works | Training data |
|---|---|---|
| **A. Rules** | Keyword lists for intent and category, plus the regex amount extractor | none |
| **B. TF-IDF + Logistic Regression** | Character n-grams (2–5) classify intent and category; regex extracts the amount | 1,599 messages |
| **C. spaCy** | NER (amount, item, merchant, date, person) + text classifier for intent, plus a separate category classifier | 1,599 messages |

The amount is always parsed by rules in `amounts.py`: digits, Arabic digits, `9.5k`, and Egyptian number words such as ميتين, ألف وخمسمية, ٢ ونص ألف, باكو (1,000) and نص أرنب (500,000). Numbers are deterministic, so a learned model gains nothing there.

## Results

Every approach was trained on the same 80% of the 2,000-message synthetic dataset. It was then tested on the held-out 20% and on 80 hand-written messages from a different author and style. Run `python -m nlp_proto.evaluate` to reproduce; the full output is in [results/results.md](results/results.md).

| Approach | Test set | Type | Amount | Category | **End-to-end** |
|---|---|---|---|---|---|
| A. Rules | synthetic (401) | 78.3% | 100% | 44.7% | **46.4%** |
| | hand-written (80) | 92.5% | 100% | 86.8% | **83.8%** |
| B. TF-IDF + LogReg | synthetic (401) | 88.0% | 100% | 79.6% | **76.8%** |
| | hand-written (80) | 100% | 100% | 92.1% | **92.5%** |
| C. spaCy | synthetic (401) | 88.8% | 99.7% | 71.1% | **72.8%** |
| | hand-written (80) | 97.5% | 100% | 80.3% | **81.2%** |

*End-to-end* = type, amount and category all correct. For a non-transaction message (greeting or question), it means the message was correctly recognised as not a transaction.

| Approach | Model size | Latency per message | Training time |
|---|---|---|---|
| A. Rules | 0 MB | 0.04 ms | – |
| B. TF-IDF + LogReg | 7.2 MB | 0.9 ms | 4.5 s |
| C. spaCy | 34.7 MB | 3.3 ms | ~2.5 min |

On the synthetic test, spaCy NER scores 81.8% F1 for item, merchant, date, person, amount and currency (exact span match).

### What the numbers say

1. **The amount is solved by rules.** They are 100% correct on both sets, including Egyptian number words and bank SMS that also show a balance.
2. **TF-IDF + logistic regression is the best classifier** for intent and category. It is also 4× smaller and faster than spaCy. With only ~1,600 examples, character n-grams beat spaCy's text classifier.
3. **spaCy is still useful for entities.** Only spaCy can return *what* was bought and *where*, which the reply and the analytics need.
4. **Rules alone do not generalise.** They are fine on the hand-written set but fall to 46% on the more varied synthetic test.

## Recommended pipeline (implemented in `pipeline.py`)

```
message → amount: rules → intent + category: TF-IDF → item/merchant/date: spaCy NER (+ merchant rule) → reply template
```

- If the amount is missing → `needs_info`: "تمام، جاكيت جديد كان بكام؟"
- The bot always picks the category itself. Below 25% confidence it uses "Other" (on the hand-written set this raised category accuracy from 92.1% to 94.7%; on the synthetic test it cost 0.9 points).
- If spaCy misses the merchant, a rule takes the place after من / من عند / عند / from / at ("من محل ابو جزم" → "ابو جزم")
- Otherwise → `confirmed`: the transaction is saved directly and the bot replies with a summary

`status` and `extracted_data` map directly to `chat_messages.extraction_status` and `chat_messages.extracted_data` in the ERD.

## Chat UI

`python -m nlp_proto.web` opens a small Arabic (RTL) chat page for trying the pipeline:

- Every reply shows the extracted type, amount, category with its confidence, item and merchant, plus the raw JSON.
- Complete messages are saved directly, with no confirmation step. They appear in the side panel with income, expense and balance totals, and each one can be deleted. They are stored in the browser's localStorage only.
- When the amount is missing, the next message is treated as the answer ("اشتريت جاكيت جديد" → "ب ١٨٥٠").

## Run it

```sh
cd ai-orchestrator/nlp_prototype
pip install -r requirements.txt
python -m nlp_proto.train            # trains on all 2,000 messages (~3 min, CPU)
python -m nlp_proto.web              # chat UI in the browser: http://localhost:8765
python -m nlp_proto.chat             # same pipeline in the terminal
python -m nlp_proto.chat "دفعت ميتين جنيه بنزين" "سلفت صاحبي ٥٠٠"
python -m nlp_proto.evaluate         # re-run the comparison
python -m unittest discover -s tests -t .
```

## Files

```
data/train/*.jsonl          2,000 synthetic labeled messages (Gemini 4 Argon, validated)
data/eval_handwritten.jsonl 80 hand-written messages, used only for testing
nlp_proto/normalize.py      Arabic digit and letter normalization
nlp_proto/amounts.py        rule-based amount extraction and number-word parser
nlp_proto/rules.py          approach A
nlp_proto/tfidf.py          approach B
nlp_proto/spacy_model.py    approach C
nlp_proto/evaluate.py       trains and compares A, B and C
nlp_proto/pipeline.py       recommended pipeline + reply templates
nlp_proto/chat.py           terminal demo
nlp_proto/web.py            local web server for the chat UI (standard library only)
web/index.html              chat UI: extracted fields, amount follow-up, saved transactions
results/                    latest comparison output
```

## Limitations (read before quoting the numbers)

- **All the data is synthetic.** The training set was generated by Gemini 4 Argon. The 80 hand-written messages came from another AI assistant, and that same assistant wrote the keyword lists for approach A, so A's hand-written score is probably optimistic. Real accuracy must be measured on real user messages collected with consent.
- Typical category mistakes: "اشتريت دوا من الصيدلية" → Shopping instead of Health, and "الحلاق ١٥٠" → a low-confidence guess. That is why the bot asks when confidence is low.
- NER misses some entities, for example "بيتزا" and "امبارح" in "طلبت بيتزا من طلبات امبارح ب 320". 98 training entities, mostly SMS dates like `10/11.` and words with a glued ل such as `لماما`, were skipped because they don't align with spaCy's tokens.
- Small classes: greeting (45), lending (41) and expense "Other" (32) need more examples.

## Next steps

1. Collect 300–500 real messages, with consent, and re-run `evaluate.py` on them.
2. Add more data for the weak categories: Health vs Shopping, and personal services.
3. Try a dialect-aware transformer (CAMeLBERT-DA or MARBERTv2) inside spaCy if NER needs to improve.
4. Optionally compare against an LLM API or TypeSafe's Jev on the same test sets, as an upper bound for accuracy and cost.
