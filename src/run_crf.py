''' August 28th, 2020
    Author: Tariq Alhindi

    Training CRF using discrete features
'''

import json
import glob
import pandas as pd
import spacy
import pickle
import os
import argparse

from utils.read_file import read_wm_essays
from model.crf import run_crf_expriments, read_features_labels


def get_sent_splits_sg(essays_dir, train_test_split_file):

	# reading essays
	train_test_split = pd.read_csv(train_test_split_file, sep=';')

	essays_txt_prg_list = []
	for file in sorted(glob.glob(essays_dir)):
	    essay = open(file).readlines()
	    essays_txt_prg_list.append(essay)

	essay_txt_str = []
	for essay in essays_txt_prg_list:
	    essay_txt_str.append(''.join(essay))

	# tokenization
	nlp = spacy.load('en_core_web_sm')

	essay_spacy = []
	for essay in essay_txt_str:
	    essay_spacy.append(nlp(essay))

	# getting sent_splits for train and test
	token_id = 0
	sent_token_ids_train, sent_token_ids_test = [], []

	for i, (doc, group) in enumerate(zip(essay_spacy, train_test_split.SET)):
	    
	    if group == "TRAIN":
	        for sent in doc.sents:
	            sent_tokens_ids = []
	            for token in sent:
	                sent_tokens_ids.append(token_id)
	                token_id +=1
	            sent_token_ids_train.append(sent_tokens_ids)
	    
	    else:
	        for sent in doc.sents:
	            sent_tokens_ids = []
	            for token in sent:
	                sent_tokens_ids.append(token_id)
	                token_id +=1
	            sent_token_ids_test.append(sent_tokens_ids)


	return sent_token_ids_train, sent_token_ids_test



def main():

	parser = argparse.ArgumentParser(description='CRF Model Training')

	# data and feature directories for the train and test sets
	parser.add_argument("--train_data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir for the training.")
	parser.add_argument("--train_feature_dir",
						default='',
						type=str,
						help="The features directory for the training data")
	parser.add_argument("--test_data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir for the test data.")
	parser.add_argument("--test_feature_dir",
						default='',
						type=str,
						help="The features directory for the training data")

	# these three parameters are only needed if the training or the test data in the SG raw format with seperate .txt and .ann files
	parser.add_argument("--train_test_split_file",
						default='',
						type=str,
						help="csv file provided by SG2017. It has two columns: essay_id, type. Type can be 'train' or 'test'")
	parser.add_argument("--train_is_sg",
						default=False,
						action='store_true',
						help="format of the training data")
	parser.add_argument("--test_is_sg",
						default=False,
						action='store_true',
						help="format of the test data")

	# features
	parser.add_argument("--features_list",
						default=[["structural_position", "structural_position_sent", "structural_punc"], ["syntactic_LCA_bin", "syntactic_POS", "syntactic_LCA_type"], ["lexsyn_1hop"],
						["structural_position", "structural_position_sent", "structural_punc", "syntactic_LCA_bin", "syntactic_POS", "syntactic_LCA_type", "lexsyn_1hop"]],
						type=list,
						help="List of list of features, each inner list is for a single expirement")

	parser.add_argument("--train_start_token_id",
                        default=0,
                        type=int,
                        help="starting token index id in the exported feature files in the training data.")
	parser.add_argument("--test_start_token_id",
                        default=80000,
                        type=int,
                        help="starting token index id in the exported feature files in the test data.")

	args = parser.parse_args()

	train_feature_dir = os.path.join(args.train_data_dir, 'features/') if args.train_feature_dir == '' else args.train_feature_dir
	test_feature_dir = os.path.join(args.test_data_dir, 'features/') if args.test_feature_dir == '' else args.test_feature_dir
	
	
	if args.train_is_sg:
		sent_token_ids_train,_ = get_sent_splits_sg(args.train_data_dir, args.train_test_split_file)
		n_train = sum([len(sent) for sent in sent_token_ids_train])
	else:
		# prepraing train features in sequence formats because extract_features.py outputs token features only
		train_essays_sent_token_label, _, _, _, _ = read_wm_essays(args.train_data_dir)

		# WM token id starts at 500000 in feature files as exported by extract_features_wm script
		token_id_train, sent_token_ids_train = args.train_start_token_id, []
		for essay in train_essays_sent_token_label:
			for sent_tokens, sent_labels in essay:
				sent_ids = []
				for token in sent_tokens:
					sent_ids.append(token_id_train)
					token_id_train += 1
				sent_token_ids_train.append(sent_ids)

		n_train = sum([len(sent) for sent in sent_token_ids_train]) # total tokens in the training data
	# pickle.dump(sent_token_ids_train, open('sent_token_ids_train.p','wb'))


	if args.test_is_sg:
		_, sent_token_ids_test = get_sent_splits_sg(args.test_data_dir, args.train_test_split_file)
		n_test = sum([len(sent) for sent in sent_token_ids_test])
	else:
		# prepraing test features in sequence formats because extract_features.py outputs token features only
		test_essays_sent_token_label, _, _, _, _ = read_wm_essays(args.test_data_dir)

		# WM token id starts at 500000 in feature files as exported by extract_features_wm script
		token_id_test, sent_token_ids_test = args.test_start_token_id, []
		for essay in test_essays_sent_token_label:
			for sent_tokens, sent_labels in essay:
				sent_ids = []
				for token in sent_tokens:
					sent_ids.append(token_id_test)
					token_id_test += 1
				sent_token_ids_test.append(sent_ids)
		n_test = sum([len(sent) for sent in sent_token_ids_test]) # total tokens in the test data


	# Starting the experiments
	for features in args.features_list:

		# preparing X_train and X_test with selected features
		X_train, y_train = read_features_labels(n_train, sent_token_ids_train, train_feature_dir, feature_files=features, suffix='.jsonlines')
		X_test, y_test = read_features_labels(n_test, sent_token_ids_test, test_feature_dir, feature_files=features, suffix='.jsonlines')
		
		# training crf model
		run_crf_expriments(features, 
							train_test_sent_splits = [sent_token_ids_train, sent_token_ids_test],
							n_train_test= [n_train, n_test],
							train_test_directories= [train_feature_dir, test_feature_dir]
						   )



if __name__ == '__main__':
    main()



