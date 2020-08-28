#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_debanjan-split_huggingface
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/wm_test/cached_*

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/transformers/examples/token-classification/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--do_predict

mv $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_SG.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/transformers/examples/token-classification/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/wm_test \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict

mv $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_WM.txt

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_debanjan-split_huggingface2
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/wm_test/cached_*

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/transformers/examples/token-classification/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--do_predict

mv $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_SG.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/transformers/examples/token-classification/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/debanjan/wm_test \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict

mv $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_WM.txt
