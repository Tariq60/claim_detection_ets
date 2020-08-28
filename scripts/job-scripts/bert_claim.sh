#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/


rm -rf /home/nlp-text/dynamic/talhindi/data/SG2017_claim/cached_*

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models/bert_claim/
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1

CUDA_VISIBLE_DEVICES=1 /home/conda/talhindi/envs/claim-detection/bin/python \
/home/nlp-text/dynamic/talhindi/src/bert/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data/SG2017_claim/ \
--labels /home/nlp-text/dynamic/talhindi/data/SG2017_claim/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_predict



rm -rf /home/nlp-text/dynamic/talhindi/data/SG2017_claim/cached_*


export MAX_LENGTH=256
export BERT_MODEL=bert-base-uncased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models/bert_claim_uncased/
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1

CUDA_VISIBLE_DEVICES=1 /home/conda/talhindi/envs/claim-detection/bin/python \
/home/nlp-text/dynamic/talhindi/src/bert/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data/SG2017_claim/ \
--labels /home/nlp-text/dynamic/talhindi/data/SG2017_claim/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_predict