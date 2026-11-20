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


def generate(prompt, system, timeout=40, search=False):
    """Return the model's text, or None if the LLM is not configured or the call fails.

    search=True lets Gemini ground the answer with Google Search (for current prices); the source
    sites are appended to the text.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024, "thinkingConfig": {"thinkingBudget": 0}},
    }
    if search:
        body["tools"] = [{"google_search": {}}]
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidate = data["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate["content"]["parts"]).strip()
        sites = [c.get("web", {}).get("title") for c in candidate.get("groundingMetadata", {}).get("groundingChunks", [])]
        sites = list(dict.fromkeys(s for s in sites if s))[:4]
        if text and sites:
            text += "\nالمصادر: " + "، ".join(sites)
        return text or None
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None
