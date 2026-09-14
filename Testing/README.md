# Testing README

The 6 scripts in this section were used for testing, and all produce an `_output.txt`.

## Scripts

### `compare_databases.py`

Compares and counts the faults between the two database builds.

### `check_figure_caption.py`

Prints the vision model's deception of a specific picture and what is actually on that page, to show the vision model's fault with clarity.

### `check_figure_caption_v5.py`

The same, but with the rebuilt database for comparison.

### `check_retrieval_quality.py`

How much of the expected evidence was found, and whether long documents crowd out short ones.

### `check_systems_match.py`

Checks that System 1's own search still matches the shared one used by the others, since it is separate.

## Workbooks

### `Testing Questions.xlsx`

Shows the questions used for testing, alongside what a good answer would contain, as well as the expected uncertainty.

### `testing spreadsheet.xlsx`

Shows the scoring for each system and the actual output of each system, along with marking comments.
