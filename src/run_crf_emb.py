''' August 28th, 2020
    Author: Tariq Alhindi

    Training CRF using bert embeddings and any selected discrete features
'''

import json
import glob
import pandas as pd
import spacy
import pickle
import os
import argparse

from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from utils.read_file import read_wm_essays
from model.crf import read_features_labels




def merge_features(bert_features, other_features):
    
    for sent_emb_features, sent_other_features in zip(bert_features, other_features):
        
        for word_emb_features, word_other_features in zip(sent_emb_features[:len(sent_other_features)], sent_other_features):
            word_other_features.update(word_emb_features)
        
        if len(sent_other_features) > len(sent_emb_features):
            for _ in range(len(sent_other_features)-len(sent_emb_features)):
                sent_other_features.pop()



def main():

	parser = argparse.ArgumentParser(description='CRF Model Training using bert embedddings as features with or without discrete features')

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


    # # these three parameters are only needed if the training or the test data in the SG raw format with seperate .txt and .ann files <-- not supported by this script
    # parser.add_argument("--train_test_split_file",
    #                     default='',
    #                     type=str,
    #                     help="csv file provided by SG2017. It has two columns: essay_id, type. Type can be 'train' or 'test'")
    # parser.add_argument("--train_is_sg",
    #                     default=False,
    #                     action='store_true',
    #                     help="format of the training data")
    # parser.add_argument("--test_is_sg",
    #                     default=False,
    #                     action='store_true',
    #                     help="format of the test data")
    
    # features
	parser.add_argument("--embeddings_train_file_name",
						default='embeddings',
						type=str,
						help="name of the embeddings features file name to be found in the train features directory")
	parser.add_argument("--embeddings_test_file_name",
						default='embeddings',
						type=str,
						help="name of the embeddings features file name to be found in the test features directory")
	parser.add_argument("--features_list",
						default=[["structural_position", "structural_position_sent", "structural_punc", "syntactic_LCA_bin", "syntactic_POS", "syntactic_LCA_type", "lexsyn_1hop"]],
						type=list,
						help="List of list of features, each inner list is for a single expirement to be run in addition to the embeddings. First expirement is always embeddings only")


	args = parser.parse_args()

	train_feature_dir = os.path.join(args.train_data_dir, 'features/') if args.train_feature_dir == '' else args.train_feature_dir
	test_feature_dir = os.path.join(args.test_data_dir, 'features/') if args.test_feature_dir == '' else args.test_feature_dir


	# prepraing train features in sequence formats because extract_features.py outputs token features only
	train_essays_sent_token_label, _, _, _, _ = read_wm_essays(args.train_data_dir)

	# WM token id starts at 500000 in feature files as exported by extract_features_wm script
	token_id_train, sent_token_ids_train, y_train = 500000, [], []
	for essay in train_essays_sent_token_label:
		for sent_tokens, sent_labels in essay:
			sent_ids = []
			for token in sent_tokens:
				sent_ids.append(token_id_train)
				token_id_train += 1
			sent_token_ids_train.append(sent_ids)

			# storing the labels of each sentence
			y_train.append(sent_labels)

	n_train = sum([len(sent) for sent in sent_token_ids_train]) # total tokens in the training data


	
	# prepraing test features in sequence formats because extract_features.py outputs token features only
	test_essays_sent_token_label, _, _, _, _ = read_wm_essays(args.test_data_dir)

	# WM token id starts at 500000 in feature files as exported by extract_features_wm script
	token_id_test, sent_token_ids_test, y_test = 500000, [], []
	for essay in test_essays_sent_token_label:
		for sent_tokens, sent_labels in essay:
			sent_ids = []
			for token in sent_tokens:
				sent_ids.append(token_id_test)
				token_id_test += 1
			sent_token_ids_test.append(sent_ids)

			# storing the labels of each sentence
			y_test.append(sent_labels)
	
	n_test = sum([len(sent) for sent in sent_token_ids_test]) # total tokens in the test data




	# Reading embeddings and labels (already in sent-sequences format)
	X_train_emb = pickle.load(open(os.path.join(train_feature_dir,args.embeddings_train_file_name+'.p'), 'rb'))
	X_test_emb = pickle.load(open(os.path.join(test_feature_dir,args.embeddings_test_file_name+'.p'), 'rb'))

	print('Experiment 1: CRF with bert embeddings only')
	crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
	crf.fit(X_train_emb, y_train)

	y_pred = crf.predict(X_test_emb)
	y_test_flat = [y for y_seq in y_test for y in y_seq]
	y_pred_flat = [y for y_seq in y_pred for y in y_seq]
	
	print('results using embeddings only:')
	print(classification_report(y_test_flat, y_pred_flat, digits=3))

	# Starting the experiments
	for features in args.features_list:

		# preparing X_train and X_test with selected features
		X_train, y_train = read_features_labels(n_train, sent_token_ids_train, train_feature_dir, feature_files=features, suffix='.jsonlines')
		X_test, y_test = read_features_labels(n_test, sent_token_ids_test, test_feature_dir, feature_files=features, suffix='.jsonlines')
		
		# merging embeddings features with discrete features and storing the results in X_train and X_test
		merge_features(X_train_emb, X_train)
		merge_features(X_test_emb, X_test)

		print('training crf model using embeddings and {} discrete features'.format(features))
		crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
		crf.fit(X_train, y_train)

		y_pred = crf.predict(X_test)
		y_test_flat = [y for y_seq in y_test for y in y_seq]
		y_pred_flat = [y for y_seq in y_pred for y in y_seq]
		
		print('results:')
		print(classification_report(y_test_flat, y_pred_flat, digits=3))

		# removing X_train, X_test before the next iteration to avoid potential memory collisions that might be cause by the passing-by-reference used in "merge_features" function
		del X_train
		del X_test



if __name__ == '__main__':
    main()



