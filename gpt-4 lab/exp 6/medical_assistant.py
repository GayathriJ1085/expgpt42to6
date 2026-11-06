"""
AI-Powered Medical Symptom Assistant using Ollama (Llama 3.2)

FOR EDUCATIONAL USE ONLY. This program does not diagnose anything and is not
a substitute for a doctor.

Workflow:
    User Input -> Patient Profile -> Safety Check (red flags) -> Symptom Extraction
    -> Prompt Generation -> LLM -> Structured Analysis -> Escalation Check
    -> Report + Disclaimer -> Follow-up questions -> Save Report

Setup (nothing new needed if earlier experiments worked):
    ollama pull llama3.2
    pip install ollama
    python medical_assistant.py
"""

import json
import os
import sys
from datetime import datetime

import ollama

MODEL = "llama3.2"
EMERGENCY_NUMBER = "112"        # India's national emergency number
MENTAL_HEALTH_HELPLINE = "14416"  # Tele-MANAS (India)

URGENCY_LEVELS = ["routine self-care", "see a doctor soon", "seek urgent medical care"]

DISCLAIMER = (
    "This is an informational AI analysis and not a medical diagnosis. "
    "It can be wrong. Please consult a qualified doctor for any health concern."
)

# ---------- Safety layer 1: rule-based red-flag detection (runs BEFORE the AI) ----------
RED_FLAGS = [
    "chest pain", "chest tightness", "chest pressure",
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "can't breathe", "cannot breathe", "struggling to breathe",
    "unconscious", "passed out", "fainted", "seizure",
    "severe bleeding", "vomiting blood", "coughing blood", "coughing up blood",
    "slurred speech", "face drooping", "weakness on one side", "numbness on one side",
    "sudden severe headache", "worst headache", "stiff neck",
    "swelling of the face", "swelling of the throat", "throat swelling",
    "overdose", "poisoning", "confusion",
]

SELF_HARM_PHRASES = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "hurt myself", "harm myself", "self harm", "self-harm",
]

# ---------- JSON structures the model must follow ----------
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"symptoms": {"type": "array", "items": {"type": "string"}}},
    "required": ["symptoms"],
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "possible_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why_considered": {"type": "string"},
                },
                "required": ["name", "why_considered"],
            },
        },
        "additional_information_needed": {"type": "array", "items": {"type": "string"}},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
        "self_care": {"type": "array", "items": {"type": "string"}},
        "urgency": {"type": "string", "enum": URGENCY_LEVELS},
        "medical_attention": {"type": "string"},
    },
    "required": [
        "possible_conditions", "additional_information_needed",
        "follow_up_questions", "self_care", "urgency", "medical_attention",
    ],
}

SYSTEM_PROMPT = (
    "You are a cautious medical information assistant for an educational project. "
    "You do NOT diagnose. You explain which conditions MAY be associated with symptoms.\n"
    "Rules:\n"
    "- Communicate uncertainty. Never present one condition as confirmed.\n"
    "- List at most 3 possible conditions, starting with the most common.\n"
    "- Use ONLY the information provided. Do not invent symptoms, test results or history.\n"
    "- Do not recommend prescription medicines and do not give medicine doses. "
    "General self-care such as rest and fluids is fine.\n"
    "- Be more cautious for children under 5 and adults over 65.\n"
    "- Choose the urgency level honestly. When unsure, choose the safer, more cautious level.\n"
    "- Advise professional medical evaluation if symptoms are severe, worsening, "
    "persistent or concerning.\n"
    "- Write in simple plain English."
)


# ---------- Step 1: collect the patient profile ----------
def ask(label, default):
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def collect_profile():
    print("Enter the patient details (press Enter to accept the default):\n")
    return {
        "Age": ask("Age", "22"),
        "Sex": ask("Sex", "Not specified"),
        "Symptoms": ask(
            "Describe your symptoms in your own words",
            "I have a sore throat, mild fever and cough.",
        ),
        "Duration": ask("Duration", "2 days"),
        "Severity": ask("Severity (mild/moderate/severe)", "Mild"),
        "Previous conditions": ask("Previous medical conditions", "None"),
        "Current medications": ask("Current medications", "None"),
        "Allergies": ask("Allergies", "None"),
    }


def profile_as_text(profile):
    return "\n".join(f"- {k}: {v}" for k, v in profile.items())


# ---------- Step 2: safety checks ----------
def find_red_flags(text):
    lowered = text.lower()
    return [flag for flag in RED_FLAGS if flag in lowered]


def mentions_self_harm(text):
    lowered = text.lower()
    return any(p in lowered for p in SELF_HARM_PHRASES)


def print_self_harm_message():
    print("\n" + "=" * 60)
    print("I'm really sorry you are going through this. You deserve support")
    print("right now, and this program is not able to give it.")
    print(f"- If you are in immediate danger, call {EMERGENCY_NUMBER} (India).")
    print(f"- You can call Tele-MANAS, a free mental health helpline in India: {MENTAL_HEALTH_HELPLINE}.")
    print("- Please reach out to someone you trust and tell them how you feel.")
    print("=" * 60 + "\n")


def print_emergency_banner(flags):
    print("\n" + "!" * 60)
    print("WARNING: possible emergency warning signs were mentioned:")
    for flag in flags:
        print(f"   - {flag}")
    print(f"Contact emergency services ({EMERGENCY_NUMBER}) or go to the nearest")
    print("hospital immediately. Do not wait for this analysis.")
    print("!" * 60 + "\n")


# ---------- Step 3: symptom extraction (natural language -> list) ----------
def extract_symptoms(profile):
    prompt = (
        "Extract the medical symptoms from the text below as a list of short phrases. "
        "Only include symptoms that the person actually states.\n\n"
        f"Text: {profile['Symptoms']}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=EXTRACT_SCHEMA,
        options={"temperature": 0},
    )
    return json.loads(response["message"]["content"])["symptoms"]


# ---------- Step 4: context analysis ----------
def analyze(profile, symptoms, extra_info):
    user_message = (
        "PATIENT PROFILE:\n"
        f"{profile_as_text(profile)}\n\n"
        f"REPORTED SYMPTOMS: {', '.join(symptoms) if symptoms else 'not clearly stated'}\n"
    )
    if extra_info:
        user_message += "\nADDITIONAL DETAILS PROVIDED LATER:\n"
        user_message += "\n".join(f"- {item}" for item in extra_info) + "\n"
    user_message += "\nProvide the analysis in the required structure."

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=ANALYSIS_SCHEMA,
        options={"temperature": 0.2},   # low = cautious and consistent
    )
    return json.loads(response["message"]["content"])


# ---------- Step 5: risk / escalation ----------
def apply_escalation(result, flags):
    """Rule-based warning signs always override the AI's urgency level."""
    if flags:
        result["urgency"] = "seek urgent medical care"
        result["medical_attention"] = (
            f"Warning signs were mentioned ({', '.join(flags)}). Contact emergency "
            f"services ({EMERGENCY_NUMBER}) or go to the nearest hospital now. "
            + result.get("medical_attention", "")
        )
    return result


# ---------- Step 6: structured report ----------
def bullets(items, empty="Not specified"):
    if not items:
        return f"  {empty}"
    return "\n".join(f"  \u2022 {item}" for item in items)


def format_report(profile, symptoms, result):
    conditions = result["possible_conditions"]
    numbered = "\n".join(f"  {i}. {c['name']}" for i, c in enumerate(conditions, 1)) or "  Not specified"
    why = "\n".join(f"  {c['name']}: {c['why_considered']}" for c in conditions) or "  Not specified"
    return (
        "MEDICAL SYMPTOM ANALYSIS\n"
        "--------------------------------\n"
        f"Patient: Age {profile['Age']}, Sex: {profile['Sex']}\n"
        f"Severity reported: {profile['Severity']}\n\n"
        f"Reported Symptoms:\n{bullets(symptoms)}\n\n"
        f"Duration:\n  {profile['Duration']}\n\n"
        f"Possible Conditions (not a diagnosis):\n{numbered}\n\n"
        f"Why These Were Considered:\n{why}\n\n"
        f"Additional Information Needed:\n{bullets(result['additional_information_needed'])}\n\n"
        f"General Self-care:\n{bullets(result['self_care'])}\n\n"
        f"Urgency Level:\n  {result['urgency'].upper()}\n\n"
        f"Medical Attention:\n  {result['medical_attention']}\n\n"
        f"Disclaimer:\n  {DISCLAIMER}\n"
        "--------------------------------\n"
    )


def save_report(path, round_no, report):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"===== Analysis round {round_no} ({datetime.now():%Y-%m-%d %H:%M}) =====\n")
        f.write(report + "\n")


# ---------- Main ----------
def main():
    print("=== AI-Powered Medical Symptom Assistant (Ollama + Llama 3.2) ===")
    print("For learning only. This is NOT a diagnosis or medical advice.\n")

    profile = collect_profile()
    all_text = f"{profile['Symptoms']} {profile['Previous conditions']}"

    if mentions_self_harm(all_text):
        print_self_harm_message()
        return

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"symptom_report_{datetime.now():%Y%m%d_%H%M%S}.txt")

    extra_info = []
    round_no = 1

    while True:
        flags = find_red_flags(all_text)
        if flags:
            print_emergency_banner(flags)

        print("Analysing... (please wait, do not type)\n")
        try:
            symptoms = extract_symptoms(profile)
            result = analyze(profile, symptoms, extra_info)
        except Exception as e:
            sys.exit(f"Error: {e}\nMake sure Ollama is running and '{MODEL}' is downloaded.")

        result = apply_escalation(result, flags)
        report = format_report(profile, symptoms, result)
        print(report)
        save_report(report_path, round_no, report)

        # ---------- Follow-up questions ----------
        questions = result["follow_up_questions"]
        if questions:
            print("Follow-up questions:")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")

        answer = input("\nAnswer them or add more details (press Enter to finish): ").strip()
        if not answer:
            break

        if mentions_self_harm(answer):
            print_self_harm_message()
            break

        extra_info.append(answer)
        all_text += " " + answer
        round_no += 1
        print()

    print(f"\nSaved to {report_path}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()