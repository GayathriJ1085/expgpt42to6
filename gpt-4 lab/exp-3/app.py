"""Web UI backend for the email assistant (exp-3).

Run:
    pip install flask openai
    set OPENAI_API_KEY=your_key      (PowerShell: $env:OPENAI_API_KEY="your_key")
    python app.py
Then open http://127.0.0.1:5000
"""
import json
import os
from datetime import datetime

from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
PAST_FILE = os.path.join(BASE, "past_emails.json")
LOG_FILE = os.path.join(BASE, "replies_log.md")

app = Flask(__name__)


def load_past():
    try:
        with open(PAST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def generate_reply(email_text: str) -> str:
    """Generate a reply using past emails as few-shot examples.

    >>> If you already have a function in email_assistant.py that does this,
    >>> replace the body with:
    >>>     from email_assistant import <your_function>
    >>>     return <your_function>(email_text)
    """
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    messages = [
        {
            "role": "system",
            "content": "You are a polite, concise customer-support assistant. "
            "Reply to the customer's email in the same tone and style as the examples.",
        }
    ]
    for ex in load_past():
        messages.append({"role": "user", "content": ex["email"]})
        messages.append({"role": "assistant", "content": ex["reply"]})
    messages.append({"role": "user", "content": email_text})

    resp = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.4)
    return resp.choices[0].message.content.strip()


def append_log(email_text: str, reply: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"## {stamp}\n**Email:** {email_text}\n\n**Reply:** {reply}\n\n---\n")
    return stamp


def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    entries = []
    for block in text.split("\n---\n"):
        block = block.strip()
        if not block.startswith("## ") or "**Email:**" not in block or "**Reply:**" not in block:
            continue
        head, rest = block.split("\n", 1)
        email_part, reply_part = rest.split("**Reply:**", 1)
        entries.append(
            {
                "time": head[3:].strip(),
                "email": email_part.replace("**Email:**", "").strip(),
                "reply": reply_part.strip(),
            }
        )
    return list(reversed(entries))  # newest first


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/past")
def api_past():
    return jsonify(load_past())


@app.post("/api/past")
def api_add_past():
    data = request.get_json(force=True)
    email, reply = (data.get("email") or "").strip(), (data.get("reply") or "").strip()
    if not email or not reply:
        return jsonify(error="Both email and reply are required."), 400
    past = load_past()
    past.append({"email": email, "reply": reply})
    with open(PAST_FILE, "w", encoding="utf-8") as f:
        json.dump(past, f, indent=2, ensure_ascii=False)
    return jsonify(past)


@app.get("/api/log")
def api_log():
    return jsonify(read_log())


@app.post("/api/reply")
def api_reply():
    email_text = (request.get_json(force=True).get("email") or "").strip()
    if not email_text:
        return jsonify(error="Please paste a customer email first."), 400
    try:
        reply = generate_reply(email_text)
    except Exception as e:  # surface API/key problems in the UI
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    stamp = append_log(email_text, reply)
    return jsonify(reply=reply, time=stamp)


if __name__ == "__main__":
    app.run(debug=True)