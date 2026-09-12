# system 2 is the same as system 1 however it uses a different model for generation 

import os
import time
from pathlib import Path
from openai import OpenAI

import shared_retrieval as shared
# to prevent my keys being compromised i have left them blank if you want to use this system put your own in here 
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

FRONTIER_LLM = "gpt-4.1"
# there is a hard cap on answer  length this is due to there being costs involved in this system however the lengths of the outputs should be still determined by the 150 words cap in the system prompt this is just a preventative measure 
MAX_TOKENS = 1500
AUDIT_LOG = shared.HERE / "system2_v2_audit.jsonl"

PRICE_INPUT_USD = 2.00
PRICE_OUTPUT_USD = 8.00
session_tokens = 0
session_cost = 0.0

# connecting to OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY or None, base_url=OPENAI_BASE_URL or None)

SYSTEM_PROMPT = shared.SYSTEM_PROMPT
def system2_answer(question: str):

    started = time.perf_counter()
    #  the shared retrieval file is now imported for the retrieval stage of this system
    hits, dense_ids, distances = shared.retrieve(question)
    global session_tokens, session_cost
    context = shared.build_context(hits)
    resp = oai.chat.completions.create(
        model=FRONTIER_LLM,
        max_completion_tokens=MAX_TOKENS,
        temperature=shared.TEMPERATURE,
        seed=shared.SEED,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": shared.user_message(context, question)},
        ],
    )
    answer = shared.strip_references(resp.choices[0].message.content)
    cost_usd = None
    if resp.usage:
        cost_usd = (resp.usage.prompt_tokens / 1_000_000 * PRICE_INPUT_USD
                    + resp.usage.completion_tokens / 1_000_000 * PRICE_OUTPUT_USD)
        cost_usd = round(cost_usd, 5)
        session_tokens += resp.usage.total_tokens
        session_cost += cost_usd

    shared.write_audit(AUDIT_LOG, question, answer, hits, dense_ids, distances,
                       model=FRONTIER_LLM, started=started,
                       prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                       output_tokens=resp.usage.completion_tokens if resp.usage else None,
                       extra={"cost_usd": cost_usd,
                              "session_cost_usd": round(session_cost, 5)})

    shared.show(question, hits, answer, "SYSTEM 2 (OpenAI frontier model) ANSWER")

    if resp.usage:
        print(f"\n[this question: {resp.usage.total_tokens} tokens  (~${cost_usd:.4f})]")
        print(f"[session so far: {session_tokens} tokens  (~${session_cost:.4f})]")
    print("=" * 80)

if __name__ != "__main__":
    pass
elif not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
    print("\nNo OpenAI key found. Either paste it into OPENAI_API_KEY at the top of")
    print("this file, or set it in the terminal first with:")
    print('    export OPENAI_API_KEY="sk-...your-key..."')
else:
    print("\nSystem 2 (frontier RAG) is ready.")
    print("Type a question and press Enter. Type  quit  to stop.\n")
    while True:
        q = input("Your question > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not q:
            continue
        print("\n...thinking...\n")
        system2_answer(q)
        print()
