# lightbot-grammar-induction
Analysis code for "Exploring the hierarchical structure of human plans via program generation"

# Code structure

## Generating figures from the paper

- `journal/examples.ipynb` - Generates pictures of the task, along with qualitative examples.
- `journal/predict-programs.ipynb` - Program enumeration and model fitting.
- `journal/descriptive.ipynb` - A number of more descriptive analyses.

## File listing

Here's a partial file listing, noting the files that contain the most important elements of the model.

- `experiment/data/human_raw/0.4/*.csv` has raw data from the experiment.
- `data/programs.csv` is the data in a format that is easier to work with, focusing on the data analyzed in the paper.
- `lb/search.py` has A* search.
- `lb/heuristics.py` has the heuristic based on an approach for the traveling salesman problem.
- `lb/program_generation.py` the greedy algorithm for generating programs.
- `lb/scoring.py` computes the various priors tested in the paper.
- `lb/fitting.py` has model fitting code, as well as code orchestrates the entire analysis (it runs trace search, program generation, etc).
- `lb/simple-ag.py` has a simplified reimplementation of the grammar induction prior that is compared to the results of sampling from a generative algorithm for the prior. Not used in analysis, but may be helpful for understanding the model.

# Install dependencies

```
pip install -r requirements.txt
```

# Run tests

```
make test
```
