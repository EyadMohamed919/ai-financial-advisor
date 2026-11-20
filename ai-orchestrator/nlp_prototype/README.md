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
- Item type comes from a keyword lexicon in `item_types.py` ("لاب" → إلكترونيات, "كوتشي" → أحذية, "لحمة" → بقالة). The training data has no item-type labels yet. If NER misses the item, a rule takes the words after اشتريت / جبت / طلبت.
- If spaCy misses the merchant, a rule takes the place after من / من عند / عند / from / at ("من محل ابو جزم" → "ابو جزم")
- Otherwise → `confirmed`: the transaction is saved directly and the bot replies with a summary

`status` and `extracted_data` map directly to `chat_messages.extraction_status` and `chat_messages.extracted_data` in the ERD.

## Router: what is the message for?

Before anything is extracted, `router.py` decides what the message is for. A second TF-IDF + logistic regression model does this, trained on `data/router/` and the transaction data. `orchestrator.py` then sends the message to the right component:

| Route | Example | Handled by | SRS |
|---|---|---|---|
| `transaction` | اشتريت لاب ب 5000 | transaction pipeline above (Classifier) | FR-010–FR-012 |
| `spending_query` | صرفت كام على الأكل الشهر ده؟ | computed from the user's records | FR-018 |
| `forecast` | هصرف كام الشهر ده؟ | run-rate projection; needs at least 5 records | FR-022 |
| `market` | الدولار بكام؟ / سهم CIB؟ | live exchange rates (open.er-api.com) + optional LLM for concepts; falls back cleanly when offline | FR-023, NFR-P03 |
| `advice` | احوش ازاي من مرتبي؟ | LLM with an aggregated summary only, or built-in rules without an LLM | FR-022, FR-029 |
| `manage` | امسح آخر حاجة | deletes the last entry | FR-016 |
| `greeting`, `out_of_scope` | ازيك / مين هيكسب الماتش؟ | fixed replies | – |

- Below 35% router confidence, the bot asks what the user meant.
- Market and advice replies always end with the informational-only disclaimer (FR-025).
- No trading actions are offered (SRS constraint).

Router accuracy on 403 held-out messages: **87.3%** (details in [results/router_results.md](results/router_results.md)). The weakest route is `out_of_scope` at 60%: off-topic questions are often mistaken for advice or spending queries, so it needs more examples.

**LLM:** optional. Copy `.env.example` to `.env` and put your key in `GEMINI_API_KEY=` (get one at https://aistudio.google.com/apikey); `.env` is git-ignored. Environment variables also work. Optionally set `GEMINI_MODEL` (default `gemini-2.5-flash`). Restart the server after changing the key. Without a key everything runs offline with rule-based replies. The LLM receives only per-category totals, never raw messages or merchants (FR-029). The Gemini call (`llm.py`) has not been tested yet, because no API key was available.

**Stocks, gold, inflation and interest rates** have no data source yet. The bot says so plainly instead of guessing.

## Chat UI

`python -m nlp_proto.web` opens a small Arabic (RTL) chat page for trying the pipeline:

- Every reply shows the extracted type, amount, category with its confidence, item and merchant, plus the raw JSON.
- Complete messages are saved directly, with no confirmation step. They appear in the side panel with income, expense and balance totals, and each one can be deleted. They are stored in the browser's localStorage only.
- When the amount is missing, the next message is treated as the answer ("اشتريت جاكيت جديد" → "ب ١٨٥٠").

## Run it

```sh
cd ai-orchestrator/nlp_prototype
pip install -r requirements.txt
python -m nlp_proto.train            # trains the classifier, spaCy NER and router (~3 min, CPU)
python -m nlp_proto.router           # evaluates the router on a held-out 20% (seconds)
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
nlp_proto/item_types.py     item-type lexicon + item fallback rule
data/router/*.jsonl         router training data (new messages per route + route labels for existing questions/corrections)
nlp_proto/router.py         message router (TF-IDF + logistic regression) and its evaluation
nlp_proto/orchestrator.py   routes each message to the right handler
nlp_proto/handlers.py       spending queries, forecast, market data, advice, undo
nlp_proto/llm.py            optional Gemini call (needs GEMINI_API_KEY)
scripts/validate_router_data.py  format checks for data/router
nlp_proto/chat.py           terminal demo
nlp_proto/web.py            local web server for the chat UI (standard library only)
web/index.html              chat UI: extracted fields, amount follow-up, saved transactions
results/                    latest comparison output
walkthrough.ipynb           step-by-step explanation that imports nlp_proto (executed outputs included)
nlp_prototype_full.ipynb    the whole prototype in one notebook: every module's code inside the cells, no nlp_proto imports
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
