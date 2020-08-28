#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_debanjan-split
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/cached_*

# CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_token_classification.py \
# --data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/ \
# --labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --per_device_train_batch_size $BATCH_SIZE \
# --save_steps $SAVE_STEPS \
# --seed $SEED \
# --do_train \
# --do_predict

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/wm_test \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict