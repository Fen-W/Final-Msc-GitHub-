# Systems

The four systems and how they work.

### `shared_retrieval.py`

So the retrieval is defined here and shared between systems 2, 3, 4.

Import it, so the four cannot differ in how they search. However system 1 uses the same method but it does not import this file.

### `system 1 v2.py`

The local RAG system that uses llama3.1.

### `system 2 v2.py`

The frontier one that uses gpt-4.1.

### `system 3 v2.py`

Hybrid RAG system both models draft, then gpt-4.1 merges them.

### `system 4 v2.py`

Agentic judges its own evidence and searches again if it is not enough.

### `test_questions.py`

The 20 testing question questions live here.

### `run_questions.py`

This feeds in the testing questions from the previous file one by one.

This approach was Taken so every system gets the same wording in the same order and no question is missed and there are no typos as I feared there would be if each question was entered manually.
