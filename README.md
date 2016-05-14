# [DEEP-CLICK-MODEL](https://github.com/THUIR/DEEP-CLICK-MODEL)

DEEP-CLICK-MODEL is a small set of Python scripts for the implementation of different click models.

# Models Implemented
- *RNN-MODEL*
- *DEEP-UBM-ALL*
- *DEEP-PSCM-ALL*

# Files
## README.md
This file.

# Requirements
- Theano
- Numba
- TensorFlow
- TFlearn
- Numpy
- Scipy
 
## bin/
Directory with the scripts.

## bin/K_TIME_DATA1/
Directory with the sample dataset.

# Format of the Input Data 
- train.qids.npy: each line is a query session with the query id and the results' id.
- train.questions.npy: each line is a query session with the query word list.
- train.answers.npy: each line is a query session with the results' title word list.
- train.additions.npy: each line is a query session with the prediction from UBM and PSCM model.
- emb_vectors.skip.1124.4m.10w.npy: each line is the word embedding vector for this word index.

# Usage
in bin/ : for RNN-MODEL, run python click_model_AAAI14.py; for DEEP-UBM-ALL and DEEP-PSCM-ALL, run python click_model_DEEP.py .





