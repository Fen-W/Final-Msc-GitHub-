<img width="2400" height="1216" alt="image" src="https://github.com/user-attachments/assets/e9f5e803-45e5-44e0-81cc-bf09adff164c" />
# Atmospheric Microplastics RAG 

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
