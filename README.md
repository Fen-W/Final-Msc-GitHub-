<img width="1000" height="1500" alt="image" src="https://github.com/user-attachments/assets/bb5e3244-906b-449f-ab6c-df8352d46694" />

The database is built once. database_build_v5.py reads the 113 PDFs listed in corpus.csv, describes the figures with a vision model, chunks the text and stores 14,274 passages. Around 15 hours, then never again.
test_questions.py holds the 20 questions and nothing else. Every system is asked the same wording in the same order.
run_questions.py does the feeding. It fetches the list, opens whichever system you name, and passes the questions in one at a time.
All four systems search through shared_retrieval.py. Same database, same search, same prompt, same temperature and seed. Retrieval is held constant so it can't be the reason one system beats another.
Six passages come back per question, capped at two from any one document so no single source dominates the answer.
Only the generation differs. System 1 uses local llama3.1, System 2 uses gpt-4.1, System 3 has both draft then merges them, System 4 judges its own evidence and searches again if it isn't satisfied.
Every answer is logged. One line per question in the audit trail, carrying the answer, the six passages, the model and the settings — which is what makes any result traceable back to its evidence.
