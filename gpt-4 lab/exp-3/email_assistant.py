"""
Experiment 3: Intelligent Email Assistant using Ollama

How it works:
    1. Load past customer emails and the replies that were sent.
    2. Convert every email into an embedding (a list of numbers that captures meaning).
    3. For a new email, find the most similar past emails (cosine similarity).
    4. Give those past emails + replies to Llama 3.2 as examples.
    5. Llama 3.2 writes a suitable reply for the new email.

Setup:
    ollama pull llama3.2
    ollama pull nomic-embed-text
    pip install ollama
    python email_assistant.py
"""

import json
import math
import re
import sys
from datetime import datetime

import ollama

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 2  # how many similar past emails to use as examples


# ---------- Step 1: load past interactions ----------
def load_past_emails(path="past_emails.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Step 2: similarity ----------
def embed(text):
    """Turn text into a vector using the local embedding model."""
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def keyword_similarity(a, b):
    """Simple fallback: overlap of words between two texts."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def prepare_index(past):
    """Embed all past emails once. Fall back to keywords if the model is missing."""
    try:
        for item in past:
            item["vector"] = embed(item["email"])
        return True
    except Exception as e:
        print(f"[!] Embedding model unavailable ({e}).")
        print(f"    Run 'ollama pull {EMBED_MODEL}' for better matching.")
        print("    Using simple keyword matching instead.\n")
        return False


def find_similar(new_email, past, use_embeddings, k=TOP_K):
    """Return the k most similar past interactions with their scores."""
    if use_embeddings:
        new_vec = embed(new_email)
        scored = [(cosine_similarity(new_vec, p["vector"]), p) for p in past]
    else:
        scored = [(keyword_similarity(new_email, p["email"]), p) for p in past]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


# ---------- Step 3: build prompt and generate reply ----------
def build_prompt(new_email, similar):
    examples = ""
    for i, (_, p) in enumerate(similar, 1):
        examples += (
            f"Example {i}\n"
            f"Customer email: {p['email']}\n"
            f"Our reply: {p['reply']}\n\n"
        )
    return (
        "Below are previous customer emails and the replies our support team sent.\n\n"
        f"{examples}"
        "Now write a reply to the new customer email below.\n"
        "Rules:\n"
        "- Follow the tone and style of the previous replies.\n"
        "- Be polite, professional and concise (3 to 5 sentences).\n"
        "- Address the customer's actual problem; do not copy the examples word for word.\n"
        "- Do not invent order numbers, dates or policies.\n"
        "- Output only the reply text.\n\n"
        f"New customer email: {new_email}\n"
        "Our reply:"
    )


def generate_reply(prompt):
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support email assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    reply = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        reply += piece
    print()
    return reply.strip()


def log_result(new_email, similar, reply):
    """Append every result to a log file (useful for the lab record)."""
    with open("replies_log.md", "a", encoding="utf-8") as f:
        f.write(f"## {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"**Customer email:** {new_email}\n\n")
        f.write("**Similar past emails used:**\n")
        for score, p in similar:
            f.write(f"- ({score:.2f}) {p['email']}\n")
        f.write(f"\n**AI reply:** {reply}\n\n---\n\n")


# ---------- Main loop ----------
def main():
    print("=== Intelligent Email Assistant (Ollama + Llama 3.2) ===\n")

    try:
        past = load_past_emails()
    except FileNotFoundError:
        sys.exit("past_emails.json not found. Keep it in the same folder as this script.")

    print(f"Loaded {len(past)} past interactions. Preparing search index...")
    use_embeddings = prepare_index(past)
    print("Ready.\n")

    while True:
        new_email = input("Paste customer email (or type 'quit'): ").strip()
        if new_email.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not new_email:
            continue

        try:
            similar = find_similar(new_email, past, use_embeddings)

            print("\n--- Similar past emails found ---")
            for score, p in similar:
                print(f"({score:.2f}) {p['email']}")
                print(f"       -> {p['reply']}")

            print("\n--- Suggested reply ---\n")
            reply = generate_reply(build_prompt(new_email, similar))
            log_result(new_email, similar, reply)
            print("\n(Saved to replies_log.md)\n")
        except Exception as e:
            print(f"\nError: {e}\nMake sure Ollama is running and '{CHAT_MODEL}' is downloaded.\n")


if __name__ == "__main__":
    main()