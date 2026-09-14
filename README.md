<img width="2400" height="1216" alt="image" src="https://github.com/user-attachments/assets/e9f5e803-45e5-44e0-81cc-bf09adff164c" />
# Atmospheric Microplastics RAG Benchmark

This repository contains the full pipeline used for the development and testing of four Retrieval Augmented Generation (RAG) systems.

All four systems are tested using the same 20 questions, database and retrieval approach. The aim is to keep the testing conditions as consistent as possible so that the systems can be compared fairly.

## How It Works

### Database Build

`database_build_v5.py` is used to build the database.

It processes the PDFs listed in `corpus.csv` and converts their contents into **14,274 passages** that can be searched by the RAG systems.

The full database build takes approximately **15 hours on an M2 Mac Studio with 32 GB of RAM**.

### Test Questions

`test_questions.py` contains the questions used to test the four systems.

A total of **20 questions** are used and the same questions are given to every system.

### Running the Questions

`run_questions.py` loads the questions and passes them into the selected system one at a time.

Each response is recorded so that the output from every system can be reviewed and compared.

### Retrieval

All four systems use the same underlying retrieval approach.

Systems 2, 3 and 4 use `shared_retrieval.py` directly.

System 1 contains its own implementation of the same retrieval logic. This is checked against the shared version using `check_systems_match.py`.

The initial retrieval stage uses the same:

* Database
* Embedding model
* Search method
* Passage selection rules

Six passages are returned for each question with a maximum of two passages from any one document. This reduces the chance of one large or repetitive document dominating the retrieved evidence.

## The Four Systems

The main difference between the systems is how the retrieved passages are used to produce the final answer.

### System 1

Uses a locally hosted **Llama 3.1** model.

### System 2

Uses **GPT 4.1**.

### System 3

Combines the local Llama 3.1 model and GPT 4.1.

### System 4

Uses an **agentic approach** that can assess the available evidence and perform additional retrieval when required.

## Audit Logs

Each answer is recorded in an audit trail.

There is one record for every question containing information such as:

* The question
* The final answer
* Retrieved passages
* Model information
* Generation settings
* Timing information
* Token usage
* Cost where applicable

These audit logs make it possible to trace each answer back to the evidence that was available to the system.

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

The experiment was designed to keep as many conditions as possible consistent across all four systems.

The same database is used for all four systems and exactly the same 20 questions are used for every system.

Retrieval uses the same database, embedding model, search method and passage selection rules.

Generation settings such as temperature and seed are fixed across the systems to reduce unnecessary variation between runs.

Every answer and its associated information is stored in the audit logs so that individual results can be checked later.

The scripts in the `Testing/` folder are used to identify possible sources of inconsistency including:

* Errors in figure descriptions
* Differences between database builds
* Retrieval bias towards longer documents
* Differences between System 1's retrieval code and the shared retrieval code

Together, these controls provide a consistent testing environment for comparing the four RAG systems.
