"""READ-ONLY OpenRouter probe: does the configured default model exist?

Loads OPENROUTER_API_KEY from .env, calls /api/v1/models and checks
whether the model id exists; then attempts a 5-token completion.
Prints HTTP status / error verbatim. No writes.
"""
import json
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_TO_CHECK = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3-32b",
    "openai/gpt-4o-mini",
    "baai/bge-m3",
]


def load_env_key(name: str) -> str:
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{name}=") or line.startswith(f'{name}="'):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    return ""


def http_get(url: str, key: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def main() -> int:
    key = load_env_key("OPENROUTER_API_KEY")
    print(f"key loaded: {'yes' if key else 'NO'} (len={len(key)})")

    status, body = http_get("https://openrouter.ai/api/v1/models", key)
    print(f"GET /models -> HTTP {status}")
    if status != 200:
        print(body[:500])
        return 1
    data = json.loads(body)
    ids = {m.get("id") for m in data.get("data", [])}
    print(f"total models listed: {len(ids)}")
    for m in MODELS_TO_CHECK:
        print(f"  {m}: {'EXISTS' if m in ids else 'NOT FOUND'}")

    # minimal completion probe with the suspicious model
    payload = json.dumps({
        "model": "google/gemini-3.1-flash-lite",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"completion probe -> HTTP {resp.status}")
            print(resp.read().decode("utf-8", "replace")[:500])
    except urllib.error.HTTPError as e:
        print(f"completion probe -> HTTP {e.code}")
        print(e.read().decode("utf-8", "replace")[:500])
    except Exception as e:
        print(f"completion probe -> FAILED: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
