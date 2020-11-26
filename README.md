# Bug-Report-Summarization
Code for Intent-aware Encoder Representations for Bug Report Summarization

The code will be split into two sections:

1. Data preproessing
2. Pretraining for Language Modeling
3. Finetuning for Summarization

Directory guide:
`BERT`: Code for Pretraining
`docker`: docker requirements file for summarization
`logs`: Log files for summarization
`src`: Code for Finetuning

## Data Preprocessing
Essential files:
`src/pretrain/args_info.py`: get arguments for each Bugzilla platform
`src/pretrain/bug_source.py`: argument class for products
`src/pretrain/data_source.py`: functions for sourcing data from the Bugzilla platform
`src/mining.py`: setting the arguments for each products

Steps:
1. Use `mining.py` to set the arguments for each Bugzilla platform
2. Use the `source_data()` function in `data_source.py` to source the data

## Pretraining for Language Modeling
Please follow the instructions at https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#pre-training for the pretraining procedure.
For regular BERT pretraining, please use `BERT/scripts/run_pretraining.sh`, for intent prediction pretraining, please use `BERT/scripts/run_pretraining_intent.sh`.

## Finetuning for Summarization
Please follow the instructions at https://github.com/nlpyang/PreSumm for the finetuning procedure.

## Next Steps
Here are some key areas to work on:

1. Building/Expanding the supervised dataset:
  - Labelling sentences for extractive summaries on Bug reports
  - What kind of channels can we use?
  - Amazon Mechanical Turk?
  - University students?
2. Explore/extend current methods of intent labelling
  - Unsupervised methods: Bayesian Network, CRF, HMM
  - Snorkel labelling via heuristics
  - Better heuristics on lingusitics features
3. Structural Summarisation
  - Supervised learning on the distribution of sentence intentions (need labelled intent data for each sentence in the extractive summaries)
  - Heuristics for intent labelling
  - Multiclass classification on other intent/summarisation datasets
  - We can include additional classification experiments to see if the model can predict intent labels of sentences
  
