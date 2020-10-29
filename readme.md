## Language Model Transformers as Evaluators for Open-domain Dialogues

This repository provides the resources to reproduce the results reported in the paper: "Language Model Transformers as Evaluators for Open-domain Dialogues" ([link](http://jens-lehmann.org/files/2020/coling_lm_dialogue_eval.pdf)).

These are the instructions for reproducing the results. We provide the following scripts and resources:


### Details about the contents

- `transformers_dialogue_evaluators.py`
    - the scripts compute probability scores for the ConvAI1 and ConvAI2 datasets using BERT, XLNet and GPT2
    - Depending on the available hardware the script can take a day or even longer to execute and compute the results.
    - just execute the script to obtain the results:
        - `python -u transformers_dialogue_evaluators.py`
- `convai(1|2)_results.pickle.bz2` - we provide the already computed probability scores as a shortcut for the correlation analysis
- `convai(1|2)_corr.ipynb` - Jupyter notebooks that:
    - calculate the various aggregated scores for dialogues
    - compute the correlation scores
    - visualize them in an interactive spreadsheet


### Instructions

Python 3.6 is used to run the scripts. We recommend using a virtual environment like (Ana|Mini)conda. Steps:

1. Install dependencies
    - `pip install jupyter requests numpy scipy scikit-learn seaborn tqdm torch==1.3.1 transformers==2.2.1 pandas qgrid`
2. Activate qgrid Jupyter extension
    - `jupyter nbextension enable --py --sys-prefix qgrid`
    - Skipping this step would prevent Jupyter from rendering an interactive spreadsheet with the correlation scores
3. Start Jupyter:
    - `jupyter notebook`
4. Open and run all the cells in the notebooks
    - the correlation scores should be computed and visualized
    - sample dialogues used in the paper are shown
