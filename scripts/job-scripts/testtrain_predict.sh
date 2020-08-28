#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_traintest/
export BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=1 /home/conda/talhindi/envs/claim-detection/bin/python \
/home/nlp-text/dynamic/talhindi/src/run_ner.py \
--data_dir /home/research/interns/talhindi/talhindi/data_newsplit/debanjan/ \
--labels /home/research/interns/talhindi/talhindi/data_newsplit/debanjan/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_eval
