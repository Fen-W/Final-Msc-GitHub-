#system 3 is the same as system 1 and 2 however it uses both models
# both modesl draft tand then the forntier model then write a final asnwer from oth drafts 

import os                      
import time
from pathlib import Path
import ollama
from openai import OpenAI

import shared_retrieval as shared

OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

LOCAL_LLM = "llama3.1:8b"
FRONTIER_LLM = "gpt-4.1"
MAX_TOKENS = 1500
AUDIT_LOG = shared.HERE / "system3_v2_audit.jsonl"

PRICE_INPUT_USD = 2.00
PRICE_OUTPUT_USD = 8.00
session_tokens = 0
session_cost = 0.0

oai = OpenAI(api_key=OPENAI_API_KEY or None, base_url=OPENAI_BASE_URL or None)

SYSTEM_PROMPT = shared.SYSTEM_PROMPT
# the prompt has to be difffernt for the draft as its a mergging the two draft together 
MERGE_PROMPT = (
    "You are given the SOURCES, and TWO draft answers written separately by two "
    "different AI systems from those same sources. Write ONE best final answer in no "
    "more than 150 words. Use ONLY claims that the sources support. Where the two "
    "drafts disagree, check the sources and keep only what they back up. Remove "
    "anything not supported. Synthesise and compare the evidence, including "
    "agreements, disagreements, limitations and gaps. Cite every factual claim with "
    "its exact chunk ID, for example [DOC017::chunk::24]. Clearly separate scientific "
    "evidence from possible policy implications, and do not rank sources or recommend "
    "policies unless directly supported. If the sources are not enough to answer, say "
    "so plainly. Final decisions remain with the human policymaker."
)
# each model is given a wraper 
def ask_local(system: str, user: str) -> str:
    """Ask the LOCAL model a question and return its written answer (this is free).

    temperature and seed are fixed to the shared values, so this draft is produced
    under exactly the same conditions as a System 1 answer."""
    resp = ollama.chat(model=LOCAL_LLM, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], options={"temperature": shared.TEMPERATURE, "seed": shared.SEED})
    return resp["message"]["content"]

def ask_frontier(system: str, user: str) -> str:
    """Ask the PAID OpenAI model, add the cost to the running totals, and return its answer."""
    global session_tokens, session_cost
    resp = oai.chat.completions.create(
        model=FRONTIER_LLM, max_completion_tokens=MAX_TOKENS,
        temperature=shared.TEMPERATURE,
        seed=shared.SEED,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    if resp.usage:
        cost_usd = (resp.usage.prompt_tokens / 1_000_000 * PRICE_INPUT_USD
                    + resp.usage.completion_tokens / 1_000_000 * PRICE_OUTPUT_USD)
        session_tokens += resp.usage.total_tokens
        session_cost += cost_usd
    return resp.choices[0].message.content
#oth models draft there answer sepratly 
def system3_answer(question: str):
    tokens_before, cost_before = session_tokens, session_cost

    started = time.perf_counter()
    hits, dense_ids, distances = shared.retrieve(question)
    context = shared.build_context(hits)
    ask = shared.user_message(context, question)

    print("   [1/3] local model drafting an answer...")
    draft_local = ask_local(SYSTEM_PROMPT, ask)
    
    print("   [2/3] frontier model drafting an answer...")
    draft_frontier = ask_frontier(SYSTEM_PROMPT, ask)

    print("   [3/3] frontier model merging and verifying the two drafts...")
    merge_input = (
        f"SOURCES:\n{context}\n\n"
        f"DRAFT A (local model):\n{draft_local}\n\n"
        f"DRAFT B (frontier model):\n{draft_frontier}\n\n"
        f"QUESTION: {question}\n\nWrite the single best final answer with citations:"
    )
    final = shared.strip_references(ask_frontier(MERGE_PROMPT, merge_input))
#the audit log saves both drafts 
    shared.write_audit(AUDIT_LOG, question, final, hits, dense_ids, distances,
                       model=f"{LOCAL_LLM} + {FRONTIER_LLM} (ensemble, merged by {FRONTIER_LLM})",
                       started=started,
                       extra={"draft_local": draft_local,
                              "draft_frontier": draft_frontier,
                              "invalid_citations_draft_local":
                                  shared.invalid_citations(draft_local, hits),
                              "invalid_citations_draft_frontier":
                                  shared.invalid_citations(draft_frontier, hits),
                              "paid_tokens_this_question": session_tokens - tokens_before,
                              "cost_usd": round(session_cost - cost_before, 5),
                              "session_cost_usd": round(session_cost, 5)})

    shared.show(question, hits, final, "SYSTEM 3 (hybrid) FINAL ANSWER")

    this_tokens = session_tokens - tokens_before
    this_cost = session_cost - cost_before
    print(f"\n[this question: {this_tokens} paid tokens  (~${this_cost:.4f})]")
    print(f"[session so far: {session_tokens} paid tokens  (~${session_cost:.4f})]")
    print("=" * 80)

if __name__ != "__main__":
    pass
elif not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
    print("\nNo OpenAI key found. Set it in the terminal first with:")
    print('    export OPENAI_API_KEY="sk-...your-key..."')
else:
    print("\nSystem 3 (hybrid RAG) is ready.")
    print("Type a question and press Enter. Type  quit  to stop.\n")
    while True:
        q = input("Your question > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not q:
            continue
        print("\n...working (this runs three steps, so it takes a little longer)...\n")
        system3_answer(q)
        print()
