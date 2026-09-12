# Local system  
from pathlib import Path
import csv, json, re, time, uuid
import numpy as np
from rank_bm25 import BM25Okapi
import ollama
import chromadb
# the retrieval is the same shared_retrieval.py, not imported from it.
# System 1 was built before that shared file existed, so it contains the same copy of 
# the search and retrieval code.
# however  Systems 2, 3 and 4 import the shared file instead.

BASE = Path("/Users/woolf/Desktop/dissertation")
HERE = Path(__file__).resolve().parent
CHROMA_DIR = BASE / "build_v5" / "vectorised_database_v5"
COLLECTION = "microplastics_corpus_v5"
EMBED_MODEL = "bge-m3"
LOCAL_LLM = "llama3.1:8b"
TOP_K = 6
MAX_PER_DOC, AUDIT_LOG = 2, HERE / "system1_v2_audit.jsonl"

col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
_all = col.get(include=["documents", "metadatas"]); _all_ids, _all_docs, _all_metas = _all["ids"], _all["documents"], _all["metadatas"]
_tokenise = lambda text: re.findall(r"\b[\wµμ.-]+\b", text.lower())
_bm25 = BM25Okapi([_tokenise(doc) for doc in _all_docs])
_by_id = dict(zip(_all_ids, zip(_all_docs, _all_metas)))
_source_details = {row["doc_id"]: row for row in csv.DictReader((BASE / "corpus" / "corpus.csv").read_text(encoding="utf-8-sig").splitlines())}

SYSTEM_PROMPT = (
    "You help environmental policymakers understand scientific evidence about "
    "atmospheric microplastics. Answer in no more than 150 words using ONLY the "
    "provided sources. Synthesise and compare relevant evidence, including agreements, "
    "disagreements, limitations and gaps. Cite every factual claim with its exact chunk "
    "ID, for example [DOC017::chunk::24]. Clearly separate scientific evidence from "
    "possible policy implications, and do not rank sources or recommend policies unless "
    "directly supported. If the evidence is insufficient, say so rather than guessing. "
    "Final decisions remain with the human policymaker."
)
# it works one question at a time with the retrieval happening for the question with the generation happening after 
# before moving onto the next question 
def system1_answer(question: str):

    started = time.perf_counter(); qe = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    res = col.query(query_embeddings=[qe], n_results=TOP_K * 3, include=["documents", "metadatas", "distances"])

    bm25_scores = _bm25.get_scores(_tokenise(question))
    dense_ids, bm25_ids = res["ids"][0], [_all_ids[i] for i in np.argsort(bm25_scores)[::-1][:TOP_K * 3] if bm25_scores[i] > 0]
    combined = {cid: (1 / (60 + dense_ids.index(cid)) if cid in dense_ids else 0) + (1 / (60 + bm25_ids.index(cid)) if cid in bm25_ids else 0) for cid in set(dense_ids + bm25_ids)}
    hits, source_counts = [], {}
    for cid in sorted(combined, key=lambda c: (-combined[c], c)):
        doc, m = _by_id[cid]; source_counts[m["doc_id"]] = source_counts.get(m["doc_id"], 0) + 1
        if source_counts[m["doc_id"]] <= MAX_PER_DOC: hits.append((cid, doc, m, combined[cid]))
        if len(hits) == TOP_K: break
# the 6 passages are put into a block of text for generation  
    context = "\n\n---\n\n".join(
        f"RETRIEVED DOCUMENT TITLE: {str(m.get('title',''))}\nCHUNK ID: [{cid}]\nPASSAGE:\n{doc}" for cid, doc, m, score in hits
    )
# the LLM is then called for generation 
    resp = ollama.chat(
        model=LOCAL_LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"SOURCES:\n{context}\n\nQUESTION: {question}\n\nAnswer in connected prose and follow the system instructions."},
        ],
        options={"temperature": 0, "seed": 11},
    )
# the reference list is the cut off at the end of generation 
    answer = re.split(r"\n\s*(?:#{1,6}\s*)?(?:References|Bibliography)\s*:?\s*\n", resp["message"]["content"], maxsplit=1, flags=re.IGNORECASE)[0].rstrip()
    citations = set(re.findall(r"\[([A-Z0-9]+::chunk::\d+)\]", answer))
    # One line saved per question, so every answer can be traced back.
    audit = {"run_id": str(uuid.uuid4()), "question": question, "answer": answer, "retrieved": [{"id": cid, "text": doc, "metadata": m, "rrf_score": score} for cid, doc, m, score in hits], "dense_distances": dict(zip(dense_ids, res["distances"][0])), "invalid_citations": sorted(citations - {cid for cid, _, _, _ in hits}), "model": LOCAL_LLM, "embedding_model": EMBED_MODEL, "temperature": 0, "seed": 11, "elapsed_seconds": round(time.perf_counter() - started, 3), "prompt_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count")}
    with AUDIT_LOG.open("a", encoding="utf-8") as f: f.write(json.dumps(audit, ensure_ascii=False) + "\n")

# the question is then printed along with answer.  this system only runs when you start the file yourself untill you close it
    print("=" * 80)
    print("QUESTION:", question)
    print("\nRETRIEVED SOURCES (the 'retrieval' that makes this RAG, not just a chatbot):")
    for cid, doc, m, score in hits:
        details = _source_details.get(m["doc_id"], {})
        print(f"  - {cid} | {details.get('authors') or 'Author not recorded'} | {str(m.get('title',''))}")
    print("\n--- SYSTEM 1 (local LLaMA-3) ANSWER ---\n")
    print(answer)
    print("=" * 80)

if __name__ == "__main__":

    print("\nSystem 1 v2 (local RAG, deterministic retrieval) is ready.")
    print("Type a question and press Enter. Type  quit  to stop.\n")

    while True:
        q = input("Your question > ").strip()

        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not q:
            continue

        print("\n...thinking (this takes ~20-40 seconds)...\n")
        system1_answer(q)
        print()
