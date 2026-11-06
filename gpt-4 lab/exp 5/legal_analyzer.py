"""
Experiment 5: Automated Legal Document Analyzer using Ollama (Llama 3.2)

Workflow:
    Legal Document -> Text Extraction -> Chunking -> Prompt Generation -> LLM
        -> Structured Extraction (parties, duration, payment, ...)
        -> Merge chunk results -> Final Report -> Save Results

Setup (nothing new needed if earlier experiments worked):
    ollama pull llama3.2
    pip install ollama

Usage:
    python legal_analyzer.py                        # analyses sample_contract.txt
    python legal_analyzer.py my_contract.txt        # analyses your own text file
    python legal_analyzer.py sample_contract.txt --chunk-size 600   # force chunking demo
"""

import argparse
import json
import os
import sys

import ollama

MODEL = "llama3.2"
DEFAULT_CHUNK_SIZE = 3000   # characters per chunk
NOT_SPECIFIED = "Not specified"

# The model must answer in exactly this JSON structure.
SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "parties": {"type": "array", "items": {"type": "string"}},
        "purpose": {"type": "string"},
        "duration": {"type": "string"},
        "key_obligations": {"type": "array", "items": {"type": "string"}},
        "payment": {"type": "string"},
        "termination": {"type": "string"},
        "important_dates": {"type": "array", "items": {"type": "string"}},
        "governing_law": {"type": "string"},
        "confidentiality": {"type": "string"},
        "key_clauses": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [
        "document_type", "parties", "purpose", "duration", "key_obligations",
        "payment", "termination", "important_dates", "governing_law",
        "confidentiality", "key_clauses", "summary",
    ],
}

# ---------- Prompt components: Role, Task, Extract, Constraint, Format ----------
ROLE = "You are a legal document analysis assistant."

TASK = "Analyze and summarize the provided document."

EXTRACT = (
    "Identify: document type, parties, purpose, obligations, payment terms, "
    "duration, termination conditions, confidentiality requirements, important "
    "dates, governing law, and potentially important clauses "
    "(such as Confidentiality, Payment, Termination, Liability, "
    "Intellectual Property, Dispute Resolution)."
)

CONSTRAINT = (
    "Use only information present in the document. Do not invent facts. "
    f"If something is not mentioned, write '{NOT_SPECIFIED}' (or return an empty list). "
    "Write the summary in simple plain English in 3 to 5 sentences."
)


def build_prompt(text, part_label=""):
    return (
        f"{ROLE}\n{TASK}\n\n{EXTRACT}\n\n{CONSTRAINT}\n\n"
        f"DOCUMENT{part_label}:\n\"\"\"\n{text}\n\"\"\""
    )


# ---------- Step 1: text extraction ----------
def read_document(path):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit("To read PDFs run: pip install pypdf  (or use a .txt file)")
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Step 2: preprocessing and chunking ----------
def preprocess(text):
    """Remove extra blank lines and trailing spaces."""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def chunk_text(text, size):
    """Split at paragraph boundaries so clauses are not cut in the middle."""
    chunks, current = [], ""
    for para in text.split("\n\n"):
        # A single paragraph longer than the limit is split by force.
        while len(para) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(para[:size])
            para = para[size:]
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------- Step 3: call the model ----------
def ask_model(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=SCHEMA,
        options={"temperature": 0.1},   # low = more factual, less creative
    )
    return json.loads(response["message"]["content"])


def analyze_chunks(chunks):
    partials = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Analysing chunk {i}/{len(chunks)} ({len(chunk)} characters)...")
        partials.append(ask_model(build_prompt(chunk, f" (PART {i} OF {len(chunks)})")))
    return partials


def merge_results(partials):
    """Combine the analysis of every chunk into one final analysis."""
    print("  Combining chunk results into a final analysis...")
    combined = json.dumps(partials, indent=2)
    prompt = (
        f"{ROLE}\n"
        "The following JSON contains analyses of different parts of ONE legal document. "
        "Merge them into a single final analysis of the whole document.\n"
        "Remove duplicates, keep every important detail, and write one overall summary.\n"
        f"{CONSTRAINT}\n\nPARTIAL ANALYSES:\n{combined}"
    )
    return ask_model(prompt)


# ---------- Step 4: format the report ----------
def as_text(value):
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else NOT_SPECIFIED
    return str(value).strip() or NOT_SPECIFIED


def bullets(items):
    if not items:
        return f"  {NOT_SPECIFIED}"
    return "\n".join(f"  \u2022 {item}" for item in items)


def format_report(d, source):
    return (
        "LEGAL DOCUMENT ANALYSIS\n"
        "--------------------------------\n"
        f"Source file: {source}\n\n"
        f"Document Type:\n  {as_text(d['document_type'])}\n\n"
        f"Parties:\n{bullets(d['parties'])}\n\n"
        f"Purpose:\n  {as_text(d['purpose'])}\n\n"
        f"Duration:\n  {as_text(d['duration'])}\n\n"
        f"Key Obligations:\n{bullets(d['key_obligations'])}\n\n"
        f"Payment:\n  {as_text(d['payment'])}\n\n"
        f"Termination:\n  {as_text(d['termination'])}\n\n"
        f"Confidentiality:\n  {as_text(d['confidentiality'])}\n\n"
        f"Governing Law:\n  {as_text(d['governing_law'])}\n\n"
        f"Important Dates:\n{bullets(d['important_dates'])}\n\n"
        f"Potentially Important Clauses:\n{bullets(d['key_clauses'])}\n\n"
        f"Summary:\n  {as_text(d['summary'])}\n\n"
        "--------------------------------\n"
        "Note: AI-generated analysis for learning purposes. "
        "It is not legal advice; verify with the original document.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Automated Legal Document Analyzer")
    parser.add_argument("file", nargs="?", default="sample_contract.txt",
                        help="path to a .txt (or .pdf) legal document")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="maximum characters per chunk (default 3000)")
    args = parser.parse_args()

    print("=== Automated Legal Document Analyzer (Ollama + Llama 3.2) ===\n")

    text = preprocess(read_document(args.file))
    print(f"Loaded '{args.file}' ({len(text)} characters).")

    chunks = chunk_text(text, args.chunk_size)
    print(f"Document split into {len(chunks)} chunk(s).\n")
    print("Analysing... (this can take a minute or two)\n")

    try:
        if len(chunks) == 1:
            result = ask_model(build_prompt(chunks[0]))
        else:
            result = merge_results(analyze_chunks(chunks))
    except Exception as e:
        sys.exit(f"Error: {e}\nMake sure Ollama is running and '{MODEL}' is downloaded.")

    report = format_report(result, os.path.basename(args.file))
    print("\n" + report)

    stem = os.path.splitext(os.path.basename(args.file))[0]
    out_file = f"analysis_{stem}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()