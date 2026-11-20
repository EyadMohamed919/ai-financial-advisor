"""Optional LLM call (Google Gemini REST API, standard library only).

Set GEMINI_API_KEY (and optionally GEMINI_MODEL) to enable it, either as environment variables or in a
`.env` file in the nlp_prototype folder (git-ignored). Without a key, callers get None and fall back to a
fixed reply, so the prototype runs fully offline.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-2.5-flash"
ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


def _load_env_file():
    """Read KEY=VALUE lines from nlp_prototype/.env. Real environment variables take priority."""
    if not os.path.exists(ENV_FILE):
        return
    with open(ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


def enabled():
    return bool(os.environ.get("GEMINI_API_KEY"))


def generate(prompt, system, timeout=20):
    """Return the model's text, or None if the LLM is not configured or the call fails."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 600},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip() or None
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None
