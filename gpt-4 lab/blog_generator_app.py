"""
Experiment 2: AI-Powered Blog Content Generator using Ollama (Llama 3.2)

Setup:
    1. Install Ollama:        https://ollama.com/download
    2. Download the model:    ollama pull llama3.2
    3. Install the library:   pip install ollama
    4. Run this script:       python blog_generator.py
"""

import re
import sys

import ollama

MODEL = "llama3.2"


def build_prompt(topic, style, tone, word_count, audience):
    """Combine the components of a good prompt into a single instruction."""
    return (
        f"Write a {word_count}-word {style.lower()} blog on {topic} "
        f"for {audience}.\n"
        f"Use simple English and a {tone.lower()} tone.\n"
        f"Include:\n"
        f"- A catchy title\n"
        f"- An introduction\n"
        f"- Key concepts or applications with real-world examples\n"
        f"- Benefits and challenges\n"
        f"- A short conclusion\n"
        f"Use headings to organise the blog."
    )


def generate_blog(prompt):
    """Send the prompt to the local Llama 3.2 model and stream the reply."""
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert blog writer who writes clear, "
                           "engaging and well-structured articles.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    blog = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)  # show text as it is generated
        blog += piece
    print()
    return blog


def save_blog(topic, text):
    """Save the blog as a Markdown file named after the topic."""
    filename = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") + "_blog.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


def ask(label, default):
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def main():
    print("=== AI-Powered Blog Content Generator (Ollama + Llama 3.2) ===\n")

    topic = ask("Topic", "Artificial Intelligence")
    style = ask("Writing style", "Professional")
    tone = ask("Tone", "Friendly")
    word_count = ask("Word count", "500")
    audience = ask("Audience", "Engineering Students")

    prompt = build_prompt(topic, style, tone, word_count, audience)
    print("\n--- Prompt sent to the model ---")
    print(prompt)
    print("\n--- Generated blog ---\n")

    try:
        blog = generate_blog(prompt)
    except ollama.ResponseError as e:
        sys.exit(f"\nModel error: {e}\nDid you run 'ollama pull {MODEL}'?")
    except Exception as e:
        sys.exit(f"\nCould not reach Ollama: {e}\nMake sure Ollama is running.")

    print(f"\nWord count (approx.): {len(blog.split())}")
    print(f"Saved to: {save_blog(topic, blog)}")


if __name__ == "__main__":
    main()