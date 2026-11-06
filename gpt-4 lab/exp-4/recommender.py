"""
Experiment 4: Personalized Recommendation System using Ollama (Llama 3.2)

Flow:
    User Profile -> Structured Prompt -> LLM Analysis -> Ranked Recommendations
                                                              |
                                             User Feedback <--+  (refine and repeat)

Setup (nothing new needed if Experiments 2 and 3 worked):
    ollama pull llama3.2
    pip install ollama
    python recommender.py
"""

import json
import sys
import textwrap
from datetime import datetime

import ollama

MODEL = "llama3.2"
NUM_RECOMMENDATIONS = 5

# The model is forced to answer in this exact JSON structure, so the output
# can be shown as a clean table instead of free text.
SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recommendation": {"type": "string"},
                    "reason": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["recommendation", "reason", "score"],
            },
        }
    },
    "required": ["recommendations"],
}

SYSTEM_PROMPT = (
    "You are an expert learning advisor for a personalized course recommendation "
    "system. You recommend courses and learning topics that suit the user's profile.\n"
    f"Always give exactly {NUM_RECOMMENDATIONS} recommendations.\n"
    "For each one give: a short recommendation title, a one-sentence reason that "
    "refers to the user's interests, level or goal, and a match score from 0 to 100.\n"
    "Order them from best match to weakest match. Respect the user's skill level "
    "and preferred learning type. Never recommend things the user asked to avoid."
)


# ---------- Step 1: collect the user profile ----------
def ask(label, default):
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def collect_profile():
    print("Enter your profile (press Enter to accept the default):\n")
    return {
        "Name": ask("Name", "Arun"),
        "Age": ask("Age", "20"),
        "Field": ask("Field of study", "Computer Science"),
        "Interests": ask("Interests", "AI, Python"),
        "Skill Level": ask("Skill level", "Beginner"),
        "Preferred Type": ask("Preferred type (practical/theoretical)", "Practical"),
        "Goal": ask("Goal", "Learn AI"),
        "About me": ask("Describe yourself in a sentence (optional)", "-"),
    }


# ---------- Step 2: convert the profile into a structured prompt ----------
def build_profile_prompt(profile):
    lines = "\n".join(f"- {k}: {v}" for k, v in profile.items() if v != "-")
    return (
        "Here is the user profile:\n"
        f"{lines}\n\n"
        f"Recommend the top {NUM_RECOMMENDATIONS} courses or learning topics for this user."
    )


# ---------- Step 3: ask the model ----------
def get_recommendations(messages):
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format=SCHEMA,              # forces valid JSON in our structure
        options={"temperature": 0.4},
    )
    content = response["message"]["content"]
    data = json.loads(content)
    recs = data["recommendations"][:NUM_RECOMMENDATIONS]
    recs.sort(key=lambda r: r["score"], reverse=True)   # best match first
    return recs, content


# ---------- Step 4: display as a table ----------
def show_table(recs):
    rank_w, rec_w, reason_w, score_w = 4, 28, 46, 6
    line = "-" * (rank_w + rec_w + reason_w + score_w + 9)
    print(line)
    print(f"{'Rank':<{rank_w}} | {'Recommendation':<{rec_w}} | "
          f"{'Reason':<{reason_w}} | {'Score':>{score_w}}")
    print(line)
    for i, r in enumerate(recs, 1):
        title = textwrap.wrap(r["recommendation"], rec_w) or [""]
        reason = textwrap.wrap(r["reason"], reason_w) or [""]
        for row in range(max(len(title), len(reason))):
            t = title[row] if row < len(title) else ""
            why = reason[row] if row < len(reason) else ""
            rank = str(i) if row == 0 else ""
            score = f"{r['score']}%" if row == 0 else ""
            print(f"{rank:<{rank_w}} | {t:<{rec_w}} | {why:<{reason_w}} | {score:>{score_w}}")
    print(line)


# ---------- Step 5: save to a log ----------
def log_round(title, recs, extra=""):
    with open("recommendations_log.md", "a", encoding="utf-8") as f:
        f.write(f"### {title} ({datetime.now():%Y-%m-%d %H:%M})\n")
        if extra:
            f.write(f"{extra}\n\n")
        f.write("| Rank | Recommendation | Reason | Score |\n|---|---|---|---|\n")
        for i, r in enumerate(recs, 1):
            f.write(f"| {i} | {r['recommendation']} | {r['reason']} | {r['score']}% |\n")
        f.write("\n")


def main():
    print("=== Personalized Recommendation System (Ollama + Llama 3.2) ===\n")

    profile = collect_profile()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_profile_prompt(profile)},
    ]

    with open("recommendations_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Profile: {profile['Name']}\n")
        for k, v in profile.items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n")

    print("\nAnalysing your profile... (this can take a little while)\n")
    try:
        recs, raw = get_recommendations(messages)
    except Exception as e:
        sys.exit(f"Error: {e}\nMake sure Ollama is running and '{MODEL}' is downloaded.")

    print(f"Personalized Recommendations for {profile['Name']}")
    print(f"Field: {profile['Field']} | Interests: {profile['Interests']} | "
          f"Level: {profile['Skill Level']}\n")
    show_table(recs)
    log_round("Initial recommendations", recs)

    # ---------- Feedback loop ----------
    round_no = 1
    while True:
        feedback = input("\nGive feedback to refine (or press Enter / type 'done' to finish): ").strip()
        if feedback.lower() in ("", "done", "quit", "exit", "q"):
            break

        # Keep the conversation history so the model remembers earlier results.
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f'The user gave this feedback: "{feedback}"\n'
                f"Update the list to exactly {NUM_RECOMMENDATIONS} recommendations that "
                "reflect this feedback while still matching the original profile."
            ),
        })

        print("\nUpdating recommendations...\n")
        try:
            recs, raw = get_recommendations(messages)
        except Exception as e:
            print(f"Error: {e}")
            continue

        round_no += 1
        print(f"Updated Recommendations (after feedback #{round_no - 1})")
        show_table(recs)
        log_round(f"Updated recommendations #{round_no - 1}", recs, f"**Feedback:** {feedback}")

    print("\nSaved to recommendations_log.md. Goodbye!")


if __name__ == "__main__":
    main()