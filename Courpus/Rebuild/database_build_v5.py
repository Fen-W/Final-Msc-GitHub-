
#
# CHOICE OF VISION MODEL
#  this script replaces the previous  LLaMA 3.2 Vision with Qwen2.5-VL. 
#  
#
# ENVIRONMENT
#   Python 3.13 in .venv. Runs entirely locally through Ollama, no external APIs.
#   Models: embeddings = bge-m3 (1024 dimensions), figures = qwen2.5vl.
#   Hardware: Mac Studio (Apple M2 Max), 32 GB unified memory.
#   The embedding model must be the same at build and query time, or the vectors
#   are not comparable.
#
# HOW TO RUN
#   ollama serve &
#   ollama pull bge-m3 ; ollama pull qwen2.5vl
#   python -u "build_v5/database_build_v5.py"                   # -u shows progress
#
#  this does take a long time to run on my mac studio m2 it took around 15 hours  
#
# OUTPUTS this file outputs.
#   vectorised_database_v5/     the database, collection "microplastics_corpus_v5"
#   parsed_document_text/       the text of each document, descriptions inlined
#   extracted_figures/          the figure images
#   figure_captions_cache.json  the descriptions, reused on a re-run
#



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

# everything loads from one folder 
BASE = Path("/Users/woolf/Desktop/dissertation")
PDF_DIR = BASE / "corpus" / "pdfs"
CSV_META = BASE / "corpus" / "corpus.csv"

WORK = BASE / "build_v5"
FIG_DIR = WORK / "extracted_figures"
PROC_DIR = WORK / "parsed_document_text"
CHROMA_DIR = WORK / "vectorised_database_v5"
# Creates the output folders 
for d in (WORK, FIG_DIR, PROC_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)


VISION_MODEL = "qwen2.5vl"

EMBED_MODEL = "bge-m3"
COLLECTION = "microplastics_corpus_v5"
# setting chunk size and so the vision model captions one at a time.
CAPTION_WORKERS = 1
IMAGE_DPI = 200
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
KEEP_ALIVE = "1h"
OLLAMA_TIMEOUT = 240


MIN_FIGURE_BYTES = 3000

MIN_CHUNK_CHARS = 100
MAX_SAME_IMAGE = 3
#fixing the issues with broken words. 

BROKEN_CHARACTERS = {
    "\ue103": "fi",
    "\ue104": "fl",
    "\ue09d": "ft",
    "\ue081": "(",
    "\ue082": ")",
    "\ue088": "-",
    "\ue089": "-",
    "\ue092": ":",
    "\ue0d0": "3",
    "\uf07c": "|",
}

#trimming the number of passages 
JUNK_PHRASES = [
    "All rights reserved",
    "A R T I C L E",
    "Start of picture text",
]


_client = ollama.Client(timeout=OLLAMA_TIMEOUT)

#using corpus.csv
def load_metadata() -> dict:

    meta = {}
    if CSV_META.exists():
        df = pd.read_csv(CSV_META).fillna("")
        for _, r in df.iterrows():
            fn = str(r.get("filename", "")).strip()
            if fn:
                meta[fn] = r.to_dict()
    return meta

META = load_metadata()
# Metadata 
def meta_for(pdf_path: Path) -> dict:

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

#caching the descriptions this was needed as i have afew builds that stopped half way through.
CAPTION_CACHE = WORK / "figure_captions_cache.json"
_caps = json.loads(CAPTION_CACHE.read_text()) if CAPTION_CACHE.exists() else {}
_cap_lock = threading.Lock()

image_sizes_seen = {}

# the vision model's prompt 
CAPTION_PROMPT = (
    "You are looking at an image taken from a scientific paper or a policy "
    "document about atmospheric microplastics.\n\n"

    "First, decide whether the image contains any information at all. If it is "
    "a logo, an icon, a page header, or a decorative graphic with no data in "
    "it, reply with the single word SKIP and write nothing else.\n\n"

    "Otherwise, describe what the image shows. Say what type of chart or image "
    "it is. Read out any text you can see, including the title, the axis "
    "labels, the units, the legend entries and the category names. If you can "
    "read the numerical values, state them. If you cannot read them, describe "
    "the shape of the data instead, such as which bar is the tallest or "
    "whether a line rises or falls across the chart.\n\n"

    "Describe only what is actually visible to you. Do not state any number, "
    "place name, date or unit that you cannot see in the image."
)


failed_captions = []


def _caption_one(img_path: Path) -> str:

    for attempt in (1, 2):

        try:
            resp = _client.chat(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": CAPTION_PROMPT,
                           "images": [str(img_path)]}],
                keep_alive=KEEP_ALIVE,
            )
            return resp["message"]["content"].strip()

        except Exception as e:
            error = e

            if attempt == 1:
                print("      retrying " + img_path.name + " after: " + str(e)[:60])

    failed_captions.append(img_path.name)
    print("      GAVE UP on " + img_path.name + " after two attempts: " + str(error)[:60])

    return "(caption failed: " + str(error) + ")"

def caption_many(paths: list[Path]) -> None:

    todo = [p for p in paths if p.name not in _caps]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as ex:
        results = list(ex.map(_caption_one, todo))
    with _cap_lock:
        for p, cap in zip(todo, results):
            _caps[p.name] = cap
        CAPTION_CACHE.write_text(json.dumps(_caps))

# 
# finds the images and notes file names.
IMG_RE = re.compile(r"!\[\]\(([^)]+)\)")

# Fixes the broken characters 
def fix_characters(text: str) -> str:

    for broken, correct in BROKEN_CHARACTERS.items():
        text = text.replace(broken, correct)


    text = re.sub("[\\ue000-\\uf8ff]", "", text)

    return text
# 
def _resolve(ref: str, fig_sub: Path) -> Path | None:
    cand = Path(ref)
    if not cand.exists():
        cand = fig_sub / Path(ref).name
    return cand if cand.exists() else None
# describing images and returning the description back to where the image was.
def parse_and_caption(pdf_path: Path) -> str:

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

    md = fix_characters(md)
    img_paths = []

    for ref in IMG_RE.findall(md):

        image = _resolve(ref, fig_sub)

        if image is None:
            continue

        size = image.stat().st_size

        if size < MIN_FIGURE_BYTES:
            continue


        image_sizes_seen[size] = image_sizes_seen.get(size, 0) + 1

        if image_sizes_seen[size] > MAX_SAME_IMAGE:
            continue

        img_paths.append(image)

    caption_many(img_paths)


    def repl(m: re.Match) -> str:
        cand = _resolve(m.group(1), fig_sub)
        if cand is None:
            return ""
        cap = _caps.get(cand.name, "SKIP")

        if cap.upper().startswith("SKIP"):
            return ""

        if cap.startswith("(caption failed"):
            return ""
        pm = re.search(r"-(\d+)-\d+\.\w+$", cand.name)
        page = f" p.{int(pm.group(1)) + 1}" if pm else ""
        return f"\n\n[FIGURE{page} — {doc_id}]: {cap}\n\n"

    md = IMG_RE.sub(repl, md)
    out_md.write_text(md)
    return md

# the chunking with some overlap added 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n[FIGURE", "\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)
# to ensure everything isn't lost at the end of a run 

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# matching by meaning 
collection = client.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)
#turning the text into numbers indexing
def embed_batch(texts: list[str]) -> list[list[float]]:

    try:
        return _client.embed(model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE)["embeddings"]
    except (AttributeError, TypeError):
        return [_client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]

def index_document(pdf_path: Path) -> int:

    m = meta_for(pdf_path)
    md = parse_and_caption(pdf_path)


    chunks = []

    for chunk in splitter.split_text(md):

        chunk = chunk.strip()

        if len(chunk) < MIN_CHUNK_CHARS:
            continue


        is_junk = False

        if len(chunk) < 400:
            for phrase in JUNK_PHRASES:
                if phrase in chunk:
                    is_junk = True

        if is_junk:
            continue

        chunks.append(chunk)

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

# checking back with corpus.csv for exclusion 
def is_excluded(pdf_path: Path) -> bool:
    status = str(META.get(pdf_path.name, {}).get("inclusion_status", "")).strip().lower()
    return status == "exclude"
#checks each document to one by one skipping the excluded oness 
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
# summarises the lost figures.
if failed_captions:
    print("")
    print(str(len(failed_captions)) + " figures could not be described and were left out:")
    for name in failed_captions:
        print("   " + name)

print("\nTotal chunks in vector DB:", collection.count())
