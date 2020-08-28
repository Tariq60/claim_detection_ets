#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/

CUDA_VISIBLE_DEVICES=1 /home/conda/talhindi/envs/flair/bin/python /home/nlp-text/dynamic/talhindi/src/flair/flair_lstm_crf.py