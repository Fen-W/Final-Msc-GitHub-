# this script feeds my 20 question that are outinedin test question inot my system in the same ourder to insure that all the system when tested are given the sme questions exatly and htere is to typos ect ect for repoducibilty

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import test_questions


HERE = Path(__file__).resolve().parent
PAUSE = 2         
RETRIES = 1       


SYSTEMS = {
    "1": ("system 1 v2.py", "system1_answer", "System 1 (local, free)"),
    "2": ("system 2 v2.py", "system2_answer", "System 2 (frontier)"),
    "3": ("system 3 v2.py", "system3_answer", "System 3 (hybrid ensemble)"),
    "4": ("system 4 v2.py", "system4_answer", "System 4 (agentic)"),
}


def load(filename):
    """Load a system file by path, because the names have spaces in them."""
    spec = importlib.util.spec_from_file_location("system_under_test", HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




if len(sys.argv) < 2 or sys.argv[1] not in SYSTEMS:
    print("\nSay which system to run:  python run_questions.py 2")
    print("  --one    ask a single question first, as a cheap test")
    print("  --fresh  clear that system's audit log before starting\n")
    raise SystemExit(1)

choice = sys.argv[1]
just_one = "--one" in sys.argv
filename, function_name, label = SYSTEMS[choice]


# start from a clean audit log, if asked 


if "--fresh" in sys.argv:
    log = HERE / ("system%s_v2_audit.jsonl" % choice)
    if log.exists():
        old_lines = len([l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()])
        log.unlink()
        print("Cleared %s (%d old record(s) removed)." % (log.name, old_lines))
    else:
        print("No existing audit log to clear.")


questions = test_questions.QUESTIONS
if just_one:
    questions = questions[:1]




print("")
print("Running %d question(s) through %s." % (len(questions), label))
print("Answers are appended to the audit log as each one finishes.")
print("")

system = load(filename)
answer = getattr(system, function_name)

began = time.perf_counter()
failed = []

for number, question in enumerate(questions, start=1):

    print("=" * 78)
    print("[%2d/%d] %s" % (number, len(questions), question[:66]))
    started = time.perf_counter()

    for attempt in range(RETRIES + 1):
        try:
            answer(question)
            print("        done in %.1f seconds" % (time.perf_counter() - started))
            break
        except Exception:
            if attempt < RETRIES:
                print("        failed, trying once more in 5 seconds...")
                time.sleep(5)
            else:
                print("        FAILED, moving on:")
                traceback.print_exc()
                failed.append("Q%02d" % number)

    print("")
    if number < len(questions):
        time.sleep(PAUSE)

print("=" * 78)
print("Finished %d question(s) in %.1f minutes." % (len(questions), (time.perf_counter() - began) / 60))

if hasattr(system, "session_cost"):
    print("Total spend this run: about $%.2f" % system.session_cost)

if failed:
    print("These questions failed and need re-running: %s" % ", ".join(failed))
else:
    print("No failures.")
print("")
