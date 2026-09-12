# system 4 agentic     
#   1. SEARCHES the database,
#   2. JUDGES whether the evidence it found is enough to answer,
#   3. if NOT enough its search and searches again ,
#   4. once satisfied (or after a set number of tries), it WRITES the final answer.
#GPT 4.1 does teh thinking and jsudges and it uterlises LangGraph for the egentic aspect 

import os, json, time
from pathlib import Path
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

import shared_retrieval as shared

OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

AGENT_LLM = "gpt-4.1"
MAX_LOOPS = 3
MAX_TOKENS = 1500
AUDIT_LOG = shared.HERE / "system4_v2_audit.jsonl"

PRICE_INPUT_USD = 2.00
PRICE_OUTPUT_USD = 8.00
session_tokens = 0
session_cost = 0.0

oai = OpenAI(api_key=OPENAI_API_KEY or None, base_url=OPENAI_BASE_URL or None)

ANSWER_PROMPT = shared.SYSTEM_PROMPT

def ask_frontier(system: str, user: str, max_tokens: int) -> str:
    global session_tokens, session_cost
    resp = oai.chat.completions.create(
        model=AGENT_LLM, max_completion_tokens=max_tokens,
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

class State(TypedDict):
    question: str
    query: str
    docs: list
    loops: int
    enough: bool
    answer: str
    dense_ids: list
    distances: list
    parse_failures: int
    judgements: list

#the 3 stages the search the judge and the subsequent answer 
def retrieve_node(state: State):
    """SEARCH step: search the database with the current wording and add any new passages.

    The search itself is the shared one, identical to Systems 1, 2 and 3. What is
    different about System 4 is that this step can run more than once."""
    hits, dense_ids, distances = shared.retrieve(state["query"])
    already = {cid for cid, _, _, _ in state["docs"]}
    new = [h for h in hits if h[0] not in already]
    print(f"   [search #{state['loops'] + 1}] '{state['query'][:60]}' -> {len(new)} new passages")
    return {"docs": state["docs"] + new, "loops": state["loops"] + 1,
            "dense_ids": dense_ids, "distances": distances}

def grade_node(state: State):
    """JUDGE step: the agent decides if the evidence is enough, or writes a better search."""
    evidence = "\n\n".join(f"[{cid}] {t[:300]}" for cid, t, m, s in state["docs"])
    decision = ask_frontier(
        "You decide whether the evidence is enough to answer a question about atmospheric "
        'microplastics. Reply ONLY as JSON: {"sufficient": true or false, '
        '"refined_query": "a better search phrase if it is not sufficient"}.',
        f"QUESTION: {state['question']}\n\nEVIDENCE:\n{evidence}",
        max_tokens=150,
    )
    parse_failed = False
    try:
        parsed = json.loads(decision[decision.find("{"): decision.rfind("}") + 1])
    except Exception:
        parsed = {"sufficient": True}
        parse_failed = True

    enough = bool(parsed.get("sufficient", True))
    refined = parsed.get("refined_query") or state["question"]

    if parse_failed:
        print("   [judge] REPLY COULD NOT BE READ -> treated as 'enough' (recorded in the audit log)")
    else:
        print("   [judge]", "evidence is enough -> will answer" if enough
              else f"not enough -> new search: '{refined[:50]}'")

    return {"enough": enough, "query": refined,
            "parse_failures": state.get("parse_failures", 0) + (1 if parse_failed else 0),
            "judgements": state.get("judgements", []) + [
                {"loop": state["loops"], "sufficient": enough,
                 "parse_failed": parse_failed,
                 "refined_query": None if enough else refined}]}

def generate_node(state: State):
    """ANSWER step: write the final, cited answer from everything gathered."""
    context = shared.build_context(state["docs"])
    ans = ask_frontier(ANSWER_PROMPT,
                       shared.user_message(context, state["question"]),
                       max_tokens=MAX_TOKENS)
    return {"answer": shared.strip_references(ans)}
#the agentic part so searching again or stopping 
def decide_next(state: State):
    """THE AGENTIC DECISION: after judging, either loop back to search, or go and answer."""
    if state["enough"] or state["loops"] >= MAX_LOOPS:
        return "generate"
    return "retrieve"
#so the search always leads back to the judge again and again until there is an answer or the cap is reached 
builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", decide_next, {"retrieve": "retrieve", "generate": "generate"})
builder.add_edge("generate", END)
agent = builder.compile()

def system4_answer(question: str):
    started = time.perf_counter()
    tokens_before, cost_before = session_tokens, session_cost
    start_state = {"question": question, "query": question,
                   "docs": [], "loops": 0, "enough": False, "answer": "",
                   "dense_ids": [], "distances": [],
                   "parse_failures": 0, "judgements": []}
    final = agent.invoke(start_state)
#the audit log now records the amount of searches and the judges decisions 
    shared.write_audit(AUDIT_LOG, question, final["answer"], final["docs"],
                       final["dense_ids"], final["distances"],
                       model=AGENT_LLM, started=started,
                       extra={"search_loops": final["loops"],
                              "passages_gathered": len(final["docs"]),
                              "judge_parse_failures": final.get("parse_failures", 0),
                              "judgements": final.get("judgements", []),
                              "paid_tokens_this_question": session_tokens - tokens_before,
                              "cost_usd": round(session_cost - cost_before, 5),
                              "session_cost_usd": round(session_cost, 5)})

    print(f"\n(The agent searched {final['loops']} time(s) and gathered "
          f"{len(final['docs'])} passages before answering.)")
    shared.show(question, final["docs"], final["answer"], "SYSTEM 4 (agentic) FINAL ANSWER")

    print(f"\n[this question: {session_tokens - tokens_before} paid tokens  (~${session_cost - cost_before:.4f})]")
    print(f"[session so far: {session_tokens} paid tokens  (~${session_cost:.4f})]")
    print("=" * 80)

if __name__ != "__main__":
    pass
elif not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
    print("\nNo OpenAI key found. Set it in the terminal first with:")
    print('    export OPENAI_API_KEY="sk-...your-key..."')
else:
    print("\nSystem 4 (agentic RAG) is ready.")
    print("Type a question and press Enter. Type  quit  to stop.\n")
    while True:
        q = input("Your question > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not q:
            continue
        print("\n...the agent is working (it may search more than once)...\n")
        system4_answer(q)
        print()
