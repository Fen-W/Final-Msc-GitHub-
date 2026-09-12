# this is my original build for the vectorised database from the corpus csv this is the one that 
# i originally planned where i found multiple issues this is being provided for context to see what 
# i have changed in the build script that i actually used the v5 one 


from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import re
import json
import pandas as pd
import pymupdf4llm
import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE = Path("/Users/woolf/Desktop/dissertation")
PDF_DIR = BASE / "corpus" / "pdfs"
CSV_META = BASE / "corpus" / "corpus.csv"

WORK = BASE / "build_v2"
FIG_DIR = WORK / "extracted_figures"
PROC_DIR = WORK / "parsed_document_text"
CHROMA_DIR = WORK / "vectorised_database"

for d in (WORK, FIG_DIR, PROC_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# The models used 
VISION_MODEL = "llama3.2-vision"
EMBED_MODEL = "bge-m3"
COLLECTION = "microplastics_corpus"

# Settings for the build.
CAPTION_WORKERS = 1
IMAGE_DPI = 200
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
KEEP_ALIVE = "1h"
OLLAMA_TIMEOUT = 180
MIN_FIGURE_BYTES = 3000

# Connecting to ollama 
_client = ollama.Client(timeout=OLLAMA_TIMEOUT)


# incorporating corpus csv
def load_metadata() -> dict:
    """Read corpus.csv into a dictionary keyed by the PDF's filename.

    Returns e.g. {"DOC001.pdf": {"doc_id": "DOC001", "title": "...", ...}, ...}
    If the CSV is missing or a row is blank, we simply skip it — the pipeline
    still works, it just has less metadata for those documents.
    """
    meta = {}
    if CSV_META.exists():
        df = pd.read_csv(CSV_META).fillna("")
        for _, r in df.iterrows():
            fn = str(r.get("filename", "")).strip()
            if fn:
                meta[fn] = r.to_dict()
    return meta

META = load_metadata()

# storing the details .
def meta_for(pdf_path: Path) -> dict:
    """Build the clean metadata record we attach to every chunk of one PDF.

    Looks up the PDF in the corpus.csv data; if a field is missing it falls back
    to a sensible default (e.g. the filename becomes the doc_id). Chroma only
    accepts str/int/float/bool in metadata, so we coerce everything to those.
    """
    row = META.get(pdf_path.name, {})
    year_raw = str(row.get("year", "")).strip()
    return {
        "doc_id": str(row.get("doc_id") or pdf_path.stem),
        "title": str(row.get("title") or pdf_path.stem),
        "year": int(year_raw) if year_raw.isdigit() else 0,
        "source_type": str(row.get("source_type") or "unknown"),
        "doc_category": str(row.get("doc_category") or "unknown"),
        "topic_tag": str(row.get("topic_tag") or "unknown"),
    }


 #saving the figure descriptions 
CAPTION_CACHE = WORK / "figure_captions_cache.json"
_caps = json.loads(CAPTION_CACHE.read_text()) if CAPTION_CACHE.exists() else {}
_cap_lock = threading.Lock()
 
#vision models prompt 
CAPTION_PROMPT = (
    "This image is a figure, chart, or table from a scientific paper on "
    "atmospheric microplastics. In 2-4 factual sentences describe: the type of "
    "chart, the variables/axes, any locations, quantities or units shown, the "
    "trend, and the single key finding. If it is a logo, journal header, or "
    "decorative image with no data, reply with exactly: SKIP"
)


def _caption_one(img_path: Path) -> str:
    """Send ONE image to the local vision model and return its text description.

    _client.chat() runs the model on your GPU. We pass the image via `images=[...]`
    and get back the model's written answer. If anything fails we return an error
    string (so one bad image can't crash the whole build).
    """
    try:
        resp = _client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": CAPTION_PROMPT,
                       "images": [str(img_path)]}],
            keep_alive=KEEP_ALIVE,
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"(caption failed: {e})"

# checking whats not been done 
def caption_many(paths: list[Path]) -> None:
    """Caption a list of images, running several at once for speed.

    - Skips any image already in the cache.
    - ThreadPoolExecutor runs CAPTION_WORKERS captions in parallel.
    - After they finish, we write them all into the cache file under a lock.
    Returns nothing — it just fills the shared `_caps` cache.
    """
    todo = [p for p in paths if p.name not in _caps]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as ex:
        results = list(ex.map(_caption_one, todo))
    with _cap_lock:
        for p, cap in zip(todo, results):
            _caps[p.name] = cap
        CAPTION_CACHE.write_text(json.dumps(_caps))



IMG_RE = re.compile(r"!\[\]\(([^)]+)\)")


def _resolve(ref: str, fig_sub: Path) -> Path | None:
    """Given an image link from the markdown, find the actual image file on disk.

    pymupdf4llm sometimes writes the path slightly differently, so we try the
    link as-is first, then look inside our figures folder. Returns the real
    Path, or None if the file genuinely isn't there (a broken link we ignore).
    """
    cand = Path(ref)
    if not cand.exists():
        cand = fig_sub / Path(ref).name
    return cand if cand.exists() else None


def parse_and_caption(pdf_path: Path) -> str:
    """Turn ONE PDF into final markdown text with figure captions inlined.

    Steps:
      1. If we already processed this PDF before, reuse the saved result (cache).
      2. pymupdf4llm.to_markdown() converts the PDF to Markdown AND saves every
         figure/table as a PNG in this document's figures sub-folder.
      3. Find all figure links, caption them all (in parallel).
      4. Replace each figure link in the text with "[FIGURE p.X — DOCID]: <caption>".
      5. Save and return the finished markdown string.
    """
    doc_id = meta_for(pdf_path)["doc_id"]
    out_md = PROC_DIR / f"{doc_id}.md"
    if out_md.exists():
        return out_md.read_text()

    fig_sub = FIG_DIR / doc_id
    fig_sub.mkdir(parents=True, exist_ok=True)

    md = pymupdf4llm.to_markdown(
        str(pdf_path),
        write_images=True,
        image_path=str(fig_sub),
        dpi=IMAGE_DPI,
    )

    img_paths = [p for ref in IMG_RE.findall(md)
                 if (p := _resolve(ref, fig_sub)) and p.stat().st_size >= MIN_FIGURE_BYTES]
    caption_many(img_paths)

    def repl(m: re.Match) -> str:
        cand = _resolve(m.group(1), fig_sub)
        if cand is None:
            return ""
        cap = _caps.get(cand.name, "SKIP")
        if cap.upper().startswith("SKIP") or cap.startswith("(caption failed"):
            return ""
        pm = re.search(r"-(\d+)-\d+\.\w+$", cand.name)
        page = f" p.{int(pm.group(1)) + 1}" if pm else ""
        return f"\n\n[FIGURE{page} — {doc_id}]: {cap}\n\n"

    md = IMG_RE.sub(repl, md)
    out_md.write_text(md)
    return md


#chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)

# Open the store on disk.
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Match pieces by meaning rather than exact words.
collection = client.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)

# embedding
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Turn a list of text chunks into a list of vectors, in one call.

    _client.embed() sends all the chunks to bge-m3 together (faster than one at a
    time). bge-m3 needs no special prefixes. Returns one list of numbers (a
    vector) per input chunk. The fallback covers older ollama versions that only
    have the single-item .embeddings() call.
    """
    try:
        return _client.embed(model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE)["embeddings"]
    except (AttributeError, TypeError):
        return [_client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


def index_document(pdf_path: Path) -> int:
    """Process ONE PDF end-to-end and add its chunks to the vector database.

    1. parse_and_caption() -> markdown text (with figure captions inlined)
    2. splitter.split_text() -> a list of chunks
    3. embed_batch() -> a vector for each chunk
    4. build an id + metadata record for each chunk
    5. collection.upsert() -> store them (upsert = insert, or overwrite if the id
       already exists, so re-running never creates duplicates)
    Returns how many chunks were added.
    """
    m = meta_for(pdf_path)
    md = parse_and_caption(pdf_path)
    chunks = [c for c in splitter.split_text(md) if c.strip()]
    if not chunks:
        return 0

    embs = embed_batch(chunks)
    ids = [f"{m['doc_id']}::chunk::{i}" for i in range(len(chunks))]
    metas = []
    for i, ch in enumerate(chunks):
        mm = dict(m)
        mm["chunk_index"] = i
        mm["has_figure"] = "[FIGURE" in ch
        metas.append(mm)

    collection.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)
    return len(chunks)

# Check corpus.csv to see if this document was marked exclude.
def is_excluded(pdf_path: Path) -> bool:
    status = str(META.get(pdf_path.name, {}).get("inclusion_status", "")).strip().lower()
    return status == "exclude"

# Go through every document skipping the excluded ones.
pdfs = sorted(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs in {PDF_DIR}")
skipped = 0
for i, pdf in enumerate(pdfs, 1):
    if is_excluded(pdf):
        skipped += 1
        print(f"[{i}/{len(pdfs)}] {pdf.name}: SKIPPED (inclusion_status=exclude)")
        continue
    n = index_document(pdf)
    print(f"[{i}/{len(pdfs)}] {pdf.name}: {n} chunks")
if skipped:
    print(f"Skipped {skipped} PDF(s) marked exclude in corpus.csv (kept in CSV, not embedded)")
print("\nTotal chunks in vector DB:", collection.count())

def search(query: str, k: int = 5, where: dict | None = None):
    """Search the database and print the top-k most relevant chunks.

    - Embed the question with the SAME model that built the DB (bge-m3).
    - collection.query() finds the k closest chunks by cosine distance
      (smaller distance = more relevant).
    - `where` optionally filters by metadata, e.g. {"source_type": "peer_reviewed"}.
    """
    qe = _client.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    res = collection.query(query_embeddings=[qe], n_results=k, where=where)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n--- {meta['doc_id']} | {str(meta.get('title',''))[:60]} | dist={dist:.3f}")
        print(doc[:400], "...")

search("What deposition rates of atmospheric microplastics were measured in urban versus rural areas?")
