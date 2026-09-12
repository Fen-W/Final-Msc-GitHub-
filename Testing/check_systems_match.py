# Proves the four systems search the database in exactly the same way.

import re
import numpy as np
import openpyxl
import ollama

import shared_retrieval as shared


BASE = shared.BASE
WORKBOOK = BASE / "testing" / "RAG_System_Testing_Workbook.xlsx"




def system1_search(question):

    qe = ollama.embeddings(model=shared.EMBED_MODEL, prompt=question)["embedding"]

    res = shared.col.query(query_embeddings=[qe], n_results=shared.TOP_K * 3,
                           include=["documents", "metadatas", "distances"])

    bm25_scores = shared._bm25.get_scores(shared._tokenise(question))

    dense_ids = res["ids"][0]
    bm25_ids = [shared._all_ids[i]
                for i in np.argsort(bm25_scores)[::-1][:shared.TOP_K * 3]
                if bm25_scores[i] > 0]

    combined = {cid: (1 / (60 + dense_ids.index(cid)) if cid in dense_ids else 0)
                     + (1 / (60 + bm25_ids.index(cid)) if cid in bm25_ids else 0)
                for cid in set(dense_ids + bm25_ids)}

    hits, source_counts = [], {}
    for cid in sorted(combined, key=lambda c: (-combined[c], c)):
        doc, m = shared._by_id[cid]
        source_counts[m["doc_id"]] = source_counts.get(m["doc_id"], 0) + 1
        if source_counts[m["doc_id"]] <= shared.MAX_PER_DOC:
            hits.append(cid)
        if len(hits) == shared.TOP_K:
            break

    return hits


# to twenty test questions from workbook

sheet = openpyxl.load_workbook(WORKBOOK, data_only=True)["System 1"]

questions = []
for row in range(6, 26):
    text = sheet.cell(row, 5).value
    if text:
        questions.append((sheet.cell(row, 1).value, text))


print("")
print("Comparing System 1's own search with the shared one used by Systems 2, 3 and 4.")
print("")

same = 0
different = 0

for name, question in questions:

    one = system1_search(question)

    hits, dense_ids, distances = shared.retrieve(question)
    other = [cid for cid, doc, m, score in hits]

    if one == other:
        same = same + 1
        print("%-5s same    %s" % (name, ", ".join(c.split("::chunk::")[0] for c in other)))
    else:
        different = different + 1
        print("%-5s DIFFERENT" % name)
        print("        system 1 gave:", one)
        print("        shared gave  :", other)

print("")
print("identical on %d of %d questions" % (same, len(questions)))

if different == 0:
    print("All four systems therefore search the database identically.")
else:
    print("WARNING: the two copies have drifted apart and must be put back in step.")
print("")
