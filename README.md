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
