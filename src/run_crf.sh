python run_crf.py \
--train_data_dir ../../final_data/tokenized_essays_split_v3/train \
--test_data_dir ../../final_data/tokenized_essays_split_v3/dev \
--features_list [['structural_position', 'structural_position_sent', 'structural_punc'], ['lexsyn_1hop'], ['syntactic_POS', 'syntactic_LCA_bin'], \
['structural_position', 'structural_position_sent', 'structural_punc', 'lexsyn_1hop', 'syntactic_POS', 'syntactic_LCA_bin']]