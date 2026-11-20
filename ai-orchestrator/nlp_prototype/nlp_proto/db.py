"""SQLite storage for the prototype: a `transactions` table shaped like the ERD's (single demo user).

The real app uses MySQL through the backend; this keeps the chat prototype stateful so the handlers
and the LLM read the user's records from a database instead of the browser.
"""
import datetime as dt
import os
import sqlite3

from .data import ROOT

DB_PATH = os.path.join(ROOT, "data", "app.db")  # git-ignored
DEMO_USER = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('income', 'expense', 'savings', 'transfer')),
    amount           REAL NOT NULL CHECK (amount > 0),
    currency         TEXT NOT NULL DEFAULT 'EGP',
    category         TEXT NOT NULL,
    item             TEXT,
    item_type        TEXT,
    merchant         TEXT,
    description      TEXT,
    source           TEXT NOT NULL DEFAULT 'chatbot',
    transaction_date TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tx_user_date ON transactions (user_id, transaction_date);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def add(data, text, user_id=DEMO_USER, date=None, source="chatbot"):
    """Save an extracted transaction (pipeline `extracted_data`). Returns the new row as a ledger dict."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO transactions (user_id, transaction_type, amount, currency, category, item, item_type, merchant,"
            " description, source, transaction_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, data["type"], data["amount"], data.get("currency") or "EGP", data["category"], data.get("item"),
             data.get("item_type_ar"), data.get("merchant"), text, source, date or dt.date.today().isoformat()))
        return get(cur.lastrowid, conn)


def get(transaction_id, conn=None):
    own = conn is None
    conn = conn or connect()
    row = conn.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)).fetchone()
    if own:
        conn.close()
    return _ledger_entry(row) if row else None


def ledger(user_id=DEMO_USER):
    """All of the user's transactions, newest first, in the shape the handlers expect."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM transactions WHERE user_id = ? ORDER BY transaction_date DESC, transaction_id DESC",
                            (user_id,)).fetchall()
    return [_ledger_entry(r) for r in rows]


def delete(transaction_id, user_id=DEMO_USER):
    with connect() as conn:  # scoped to the user, so one user can never delete another's record (FR-026)
        return conn.execute("DELETE FROM transactions WHERE transaction_id = ? AND user_id = ?",
                            (transaction_id, user_id)).rowcount > 0


def _ledger_entry(row):
    return {"id": row["transaction_id"], "type": row["transaction_type"], "amount": row["amount"],
            "currency": row["currency"], "category": row["category"], "item": row["item"],
            "item_type": row["item_type"], "merchant": row["merchant"], "date": row["transaction_date"],
            "label": row["item"] or row["merchant"] or row["category"]}
