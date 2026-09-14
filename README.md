<img width="2400" height="1216" alt="image" src="https://github.com/user-attachments/assets/e9f5e803-45e5-44e0-81cc-bf09adff164c" />

This repository is the full pipelines used for the development and the testing of the four Retrieval Augmented Generation RAG Systems.

All four systems are tested using the same 20 questions, database and retrieval. The aim is to keep as much consistency as possible.

## How it works

`database_build_v5.py` is used to build the database. It turns the PDFs that are listed in `Corpus.csv` into 14,274 passages in the database.

It takes around 15 hours on a M2 Mac Studio 32GB RAM.

`test_questions.py` is where the testing questions for the system are stored.

`run_questions.py` loads the questions into the system one by one.

All systems use the same underlying retrieval approach. Systems 2, 3 and 4 use `shared_retrieval.py` directly. System 1 contains its own implementation of the same retrieval logic which is checked against the shared version using `check_systems_match.py`.

The initial retrieval stage uses the same database, embedding model, search method and passage selection rules. Six passages are returned for each question with a maximum of two passages from any one document.

The main difference in the systems is how the retrieved passages are used to produce the final output.

**System 1** uses a locally hosted Llama 3.1 model.

**System 2** uses GPT 4.1.

**System 3** combines the two.

**System 4** uses an agentic approach.

Each answer is recorded in an audit trail. There is one record for every question containing the final answer, retrieved passages, model information and generation settings.

## What Is in This Repository

### `Corpus/`

Contains the information used to build the literature corpus and database.

### `Systems/`

Contains the four RAG systems together with the shared retrieval and testing code.

### `Testing/`

Contains scripts used to check the database, retrieval process and system results for possible faults or inconsistencies.

### `Outputs/`

Contains the audit logs produced by the four systems during testing.

## Reproducibility

The same database is used for all four systems and exactly the same 20 questions are used for every system.

Retrieval uses the same database, embedding model, search method and passage selection rules.

Generation settings such as temperature and seed are fixed across all four systems.

Every answer and its additional information is stored in the audit logs.

The scripts in the `Testing` folder are used to identify possible sources of inconsistency, such as errors in figure descriptions, differences between database builds and retrieval bias towards longer documents.

# Environment and Requirements

## Python

Everything to run the system is in `requirements.txt`:

    chromadb
    ollama
    openai
    langgraph
    rank-bm25
    pymupdf / pymupdf4llm
    langchain-text-splitters
    openpyxl
    python-docx
    pandas / numpy

Install them into a fresh virtual environment before anything is ran

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Ollama (local models)

The build and System 1 both run locally through Ollama, not through an API:

- `qwen2.5vl` — the vision model used for fig captions
- `bge-m3` — the embeddings used store and search passages
- `llama3.1` — system 1s model

Ollama needs to be installed and running, with these three models pulled, before
`database_build_v5.py` or System 1 will work

    ollama pull qwen2.5vl
    ollama pull bge-m3
    ollama pull llama3.1

## OpenAI API

Systems 2, 3 and 4 all call GPT-4.1. An OpenAI API key needs to be in place in the environment for these to you can add this to the top of each script.

## Note on paths

Some scripts (including `database_build_v5.py`) currently point at fixed folder on my desk top. They will need to be updated to match wherever this repository is run from in oder to get everything to link up together. All the but all the files needed are in the repository. It's just the paths inside a few scripts that need changing for full reproducibility.

## Running it in order

1. Set up the environment and models above.
2. Run `database_build_v5.py` once to build the vector database.
3. Run `run_questions.py`, pointing it at whichever system you want to test.
4. Run the scripts in `Testing/` against the build and the outputs to check
   for faults.
