#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_SG_WM
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/cached_*

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict

cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_all.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict

cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_SG.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert/run_ner.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict



export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_weightedloss_SG_WM
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/cached_*

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_weighted_loss_claim/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict


cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_all.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_weighted_loss_claim/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict


cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_SG.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_weighted_loss_claim/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict



export MAX_LENGTH=256
export BERT_MODEL=bert-base-uncased
export OUTPUT_DIR=/home/nlp-text/dynamic/talhindi/models2/bert_claim_imho_SG_WM
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=0
export SEED=1
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG/cached_*
rm -rf /home/nlp-text/dynamic/talhindi/data_newsplit/SG/WM_test/cached_*

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_imho/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict

cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_all.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_imho/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/SG_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict


cp $OUTPUT_DIR/test_results.txt $OUTPUT_DIR/test_results_SG.txt

CUDA_VISIBLE_DEVICES=0 /home/conda/talhindi/envs/claim-detection/bin/python /home/nlp-text/dynamic/talhindi/src/bert_imho/run_token_classification.py \
--data_dir /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/WM_test/ \
--labels /home/nlp-text/dynamic/talhindi/data_newsplit/SG_WM/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict