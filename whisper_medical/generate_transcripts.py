import os
from pathlib import Path
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are generating synthetic medical French transcripts for urgent care scenarios.
Write short clinical-style phrases in French, similar to how doctors dictate or write urgent care notes.
- Use common medical acronyms (like PEC, HD, HTA, DRA, DSM, etc.).
- Sentences should often be abbreviated, fragmented, or telegraphic (not fully grammatical).
- Cover topics like trauma, cardiac issues, respiratory problems, post-op complications, metabolic disorders, and urgent care evaluations.
- Mix single-word diagnoses and longer notes (~5-20 words).
- Use natural French medical shorthand.
Example style: "stable HD, pas de signe de DRA", "gsc 15, pas de DSM", "PEC SMUR, malaise au domicile".
"""

USER_PROMPT = """
Generate 200 French urgent care medical phrases according to the above instructions.
Vary between very short (1-2 words) and longer (~20 words) entries.
Output one phrase per line, without numbering or explanations.
"""


def generate_transcripts(output_dir: Path):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.7,
        max_tokens=3000,
    )
    corpus_text = response["choices"][0]["message"]["content"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_transcripts.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(corpus_text)

    print("Synthetic corpus saved to {output_path}")
