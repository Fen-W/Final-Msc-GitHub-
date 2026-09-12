# Counts the faults in both databases so the first build and the rebuild can be compared.



import re
import chromadb



LOST_CHARACTERS = re.compile("[-]")

# A claim such as "increased by 50%".
NUMBER_CLAIM = re.compile(r"(increased|decreased|risen) by \d+%")


def count_faults(path, name):

    collection = chromadb.PersistentClient(path=path).get_collection(name)
    data = collection.get(include=["documents", "metadatas"])

    documents = []
    short = 0
    headings = 0
    lost = 0
    invented = 0

    for passage, detail in zip(data["documents"], data["metadatas"]):

        if detail["doc_id"] not in documents:
            documents.append(detail["doc_id"])

        if len(passage) < 100:
            short = short + 1
            if passage.strip().startswith("#"):
                headings = headings + 1

        if LOST_CHARACTERS.search(passage):
            lost = lost + 1

      
        claim = NUMBER_CLAIM.search(passage)

        if claim and "[FIGURE" in passage:

            start = passage.find("[FIGURE")

            # the description ends at the next picture comment, heading or blank block
            ends = [passage.find(m, start) for m in ["<!--", "\n#", "\n\n\n"]]
            ends = [e for e in ends if e != -1]
            end = min(ends) if ends else len(passage)

            if start <= claim.start() < end:
                invented = invented + 1

    return [len(data["documents"]), len(documents), short, headings, lost, invented]


first = count_faults("/Users/woolf/Desktop/dissertation/build_v2/vectorised_database",
                     "microplastics_corpus")

rebuilt = count_faults("/Users/woolf/Desktop/dissertation/build_v5/vectorised_database_v5",
                       "microplastics_corpus_v5")


labels = ["Total passages",
          "Documents",
          "Passages under 100 characters",
          "Heading only passages",
          "Passages with lost characters",
          "Invented claims in figure descriptions"]

print("")
print("%-40s %10s %10s" % ("", "first", "rebuilt"))
print("-" * 62)

for i in range(len(labels)):
    print("%-40s %10d %10d" % (labels[i], first[i], rebuilt[i]))
