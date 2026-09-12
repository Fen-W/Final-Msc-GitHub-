# this is the shared search function that all of the systems use
# when it loads it reads all of the passages in my database and then build and index over the top of them.
# it runs two searches on that uses meaning and other the exact wording  and then merges them back together.
# the maximum passage it can take from one document is 2 and it  is capped at 6 passages per question
import csv
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi


BASE = Path("/Users/woolf/Desktop/dissertation")   # where the database and corpus live
HERE = Path(__file__).resolve().parent
CHROMA_DIR = BASE / "build_v5" / "vectorised_database_v5"
COLLECTION = "microplastics_corpus_v5"

EMBED_MODEL = "bge-m3"
TOP_K = 6
MAX_PER_DOC = 2
POOL = TOP_K * 3
TEMPERATURE = 0
SEED = 11

 # this is the system prompt although not directly used in the retrieval part it's stored here so the system prompt is then called by
  # the later systems it keeps all the prompts fair across the systems 
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

 # to access the database and read all of the passage and build the index and turn the questions into embeddings
col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)

_all = col.get(include=["documents", "metadatas"])
_all_ids, _all_docs, _all_metas = _all["ids"], _all["documents"], _all["metadatas"]

_tokenise = lambda text: re.findall(r"\b[\wµμ.-]+\b", text.lower())
_bm25 = BM25Okapi([_tokenise(doc) for doc in _all_docs])
_by_id = dict(zip(_all_ids, zip(_all_docs, _all_metas)))

_source_details = {
    row["doc_id"]: row
    for row in csv.DictReader(
        (BASE / "corpus" / "corpus.csv").read_text(encoding="utf-8-sig").splitlines()
    )
}

def retrieve(question: str, top_k: int = TOP_K):
    """Find the passages most relevant to a question.

    Two searches run side by side. The first compares meaning, by turning the question
    into numbers and finding passages whose numbers are closest. The second compares
    exact words, which catches a passage that uses the same term even when the meaning
    match misses it. Their two rankings are then merged.

    Returns a list of (chunk id, passage text, source details, score).
    """
    question_numbers = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
 #  the first search 
    res = col.query(query_embeddings=[question_numbers], n_results=POOL,
                    include=["documents", "metadatas", "distances"])
    dense_ids = res["ids"][0]

 # the bm 25 search 
    bm25_scores = _bm25.get_scores(_tokenise(question))
    bm25_ids = [_all_ids[i] for i in np.argsort(bm25_scores)[::-1][:POOL] if bm25_scores[i] > 0]
 # then combine the lists together and take the best 6 
    combined = {
        cid: (1 / (60 + dense_ids.index(cid)) if cid in dense_ids else 0)
             + (1 / (60 + bm25_ids.index(cid)) if cid in bm25_ids else 0)
        for cid in set(dense_ids + bm25_ids)
    }

    hits, source_counts = [], {}
    for cid in sorted(combined, key=lambda c: (-combined[c], c)):
        doc, m = _by_id[cid]
        source_counts[m["doc_id"]] = source_counts.get(m["doc_id"], 0) + 1
        if source_counts[m["doc_id"]] <= MAX_PER_DOC:
            hits.append((cid, doc, m, combined[cid]))
        if len(hits) == top_k:
            break

    return hits, dense_ids, res["distances"][0]


# puts  all of the 6 passages into a block of text for generation
def build_context(hits):
    """Glue the retrieved passages into one labelled block for the model to read."""
    return "\n\n---\n\n".join(
        f"RETRIEVED DOCUMENT TITLE: {str(m.get('title',''))}\nCHUNK ID: [{cid}]\nPASSAGE:\n{doc}"
        for cid, doc, m, score in hits
    )


def user_message(context: str, question: str):
    """The wording wrapped around the sources and the question. Same for every system."""
    return (f"SOURCES:\n{context}\n\nQUESTION: {question}\n\n"
            f"Answer in connected prose and follow the system instructions.")


# cutting off the references at the end of the answer
def strip_references(text: str) -> str:
    """Remove a trailing reference list, so it is not mistaken for cited evidence."""
    return re.split(r"\n\s*(?:#{1,6}\s*)?(?:References|Bibliography)\s*:?\s*\n",
                    text, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()

def find_citations(answer: str):
    """Pull every chunk id the answer claims to be citing."""
    return set(re.findall(r"\[([A-Z0-9]+::chunk::\d+)\]", answer))


# this finds any passages that are used and passage that were used that were not provided.
def invalid_citations(answer: str, hits):
    """Any chunk id the answer cites that was never actually retrieved."""
    return sorted(find_citations(answer) - {cid for cid, _, _, _ in hits})

# then to print the passages found and the answer and save a full copy of run for reproducibility 
def show(question: str, hits, answer: str, heading: str):
    """Print the question, the sources that were retrieved, and the answer."""
    print("=" * 80)
    print("QUESTION:", question)
    print("\nRETRIEVED SOURCES (the 'retrieval' that makes this RAG, not just a chatbot):")
    for cid, doc, m, score in hits:
        details = _source_details.get(m["doc_id"], {})
        print(f"  - {cid} | {details.get('authors') or 'Author not recorded'} | {str(m.get('title',''))}")
    print(f"\n--- {heading} ---\n")
    print(answer)


def write_audit(path: Path, question: str, answer: str, hits, dense_ids, distances,
                model: str, started: float, prompt_tokens=None, output_tokens=None, extra=None):
    """Save a full record of the run, so any answer in the results can be traced back."""
    record = {
        "run_id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "retrieved": [{"id": cid, "text": doc, "metadata": m, "rrf_score": score}
                      for cid, doc, m, score in hits],
        "dense_distances": dict(zip(dense_ids, distances)),
        "invalid_citations": invalid_citations(answer, hits),
        "model": model,
        "embedding_model": EMBED_MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    if extra:
        record.update(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
