#!/bin/bash
#$ -j y     # this puts stderr in stdout
#$ -o /home/research/interns/talhindi/talhindi/outputs/


cd /home/nlp-text/dynamic/talhindi/semeval2020_task11/
CUDA_VISIBLE_DEVICES=2 /home/conda/talhindi/envs/propa/bin/python -m span_identification --config configs/si_config2.yml --do_train --do_eval
CUDA_VISIBLE_DEVICES=2 /home/conda/talhindi/envs/propa/bin/python -m span_identification --config configs/si_config2.yml --do_predict

