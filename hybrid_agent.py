import os
import json
import requests

MIMO_API_KEY = os.getenv("MIMO_API_KEY")

MIMO_URL = "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
MIMO_MODEL = "mimo-v2.5-pro"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "hermes-id"

MAX_HISTORY = 4

SYSTEM_PROMPT = (
    "Kamu adalah Hermes AI Agent milik Jerry. "
    "Jawab bahasa Indonesia natural, jelas, teknis, dan langsung ke solusi. "
    "Fokus programming, API, automation, debugging, Windows, PowerShell, Python, dan local AI. "
    "Jawaban jangan panjang kecuali diminta full script."
)

mode = "auto"
messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def trim_messages():
    system = [m for m in messages if m["role"] == "system"]
    normal = [m for m in messages if m["role"] != "system"]
    return system[:1] + normal[-MAX_HISTORY:]


def ask_mimo(user_messages):
    if not MIMO_API_KEY:
        return None, "MIMO_API_KEY belum diset"

    headers = {
        "api-key": MIMO_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": MIMO_MODEL,
        "messages": user_messages,
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 384,
    }

    try:
        r = requests.post(MIMO_URL, headers=headers, json=payload, timeout=90)
    except Exception as e:
        return None, f"Request MiMo error: {e}"

    try:
        data = r.json()
    except Exception:
        return None, r.text

    if r.status_code == 200:
        return data["choices"][0]["message"]["content"], None

    if r.status_code == 429:
        return None, "MiMo kena limit 429"

    return None, json.dumps(data, indent=2, ensure_ascii=False)


def ask_ollama(user_messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": user_messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_predict": 384,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except Exception as e:
        return f"❌ Ollama error: {e}"

    try:
        data = r.json()
    except Exception:
        return f"❌ Ollama response bukan JSON:\n{r.text}"

    if "message" in data:
        return data["message"]["content"]

    return json.dumps(data, indent=2, ensure_ascii=False)


def answer(user_input):
    global messages

    messages.append({"role": "user", "content": user_input})
    compact = trim_messages()

    if mode == "local":
        ans = ask_ollama(compact)
        messages.append({"role": "assistant", "content": ans})
        return "[LOCAL hermes-id]\n" + ans

    if mode == "mimo":
        ans, err = ask_mimo(compact)
        if ans:
            messages.append({"role": "assistant", "content": ans})
            return "[MIMO]\n" + ans
        return f"❌ MiMo gagal: {err}"

    # auto mode
    ans, err = ask_mimo(compact)
    if ans:
        messages.append({"role": "assistant", "content": ans})
        return "[MIMO]\n" + ans

    fallback = ask_ollama(compact)
    messages.append({"role": "assistant", "content": fallback})
    return f"[MiMo gagal: {err}]\n[Fallback LOCAL hermes-id]\n{fallback}"


def main():
    global mode, messages

    print("=== Hybrid Hermes Agent ===")
    print("Mode default: auto")
    print("Command:")
    print("  /auto   = MiMo dulu, fallback Ollama kalau limit")
    print("  /mimo   = paksa MiMo")
    print("  /local  = paksa Ollama lokal")
    print("  /clear  = hapus context")
    print("  /exit   = keluar")
    print()

    while True:
        user_input = input("Jerry: ").strip()

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/exit", "exit", "quit", "keluar"):
            break

        if cmd == "/auto":
            mode = "auto"
            print("✅ Mode: auto")
            continue

        if cmd == "/mimo":
            mode = "mimo"
            print("✅ Mode: MiMo only")
            continue

        if cmd == "/local":
            mode = "local"
            print("✅ Mode: local Ollama")
            continue

        if cmd == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("✅ Context dibersihkan")
            continue

        print()
        print(answer(user_input))
        print()


if __name__ == "__main__":
    main()