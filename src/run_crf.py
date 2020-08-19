import json
import glob
import pandas as pd
import spacy
import pickle

from utils.read_file import read_wm_essays
from model.crf import run_crf_expriments, read_features_labels


def get_sent_splits(essays_dir = '/Users/talhindi/Documents/claim_detection/data/SG2017/*.txt',
					train_test_split_file='/Users/talhindi/Documents/claim_detection/data/SG2017/train-test-split.csv'):

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

	# print('testing on SG data')
	# n_train, n_test = 119752, 29815
	# sent_token_ids_train, _ = get_sent_splits()


	# run_crf_expriments(['structural_position', 'structural_punc', 'structural_position_sent','lexsyn_1hop'], 
	# 	train_test_sent_splits = [sent_token_ids_train, sent_token_ids_test], n_train_test= [n_train, n_test],
	# 	train_test_directories=['/Users/talhindi/Documents/claim_detection_wm/claim_features/train/',
	# 							'/Users/talhindi/Documents/claim_detection_wm/claim_features/test/'])

	
	# print('testing on WM data')
	# n_train, n_test = 119752, 27134
	# essays_sent_token_label, _, _, _, _ = read_wm_essays()

	# # WM token id starts at 500000 in feature files as exported by extract_features_wm script
	# token_id, sent_token_ids_test = 500000, []
	# for essay in essays_sent_token_label:
	# 	for sent_tokens, sent_labels in essay:
	# 		sent_ids = []
	# 		for token in sent_tokens:
	# 			sent_ids.append(token_id)
	# 			token_id += 1
	# 		sent_token_ids_test.append(sent_ids)

	# run_crf_expriments(['structural_position', 'structural_punc', 'structural_position_sent','lexsyn_1hop'], 
	# 	train_test_sent_splits = [sent_token_ids_train, sent_token_ids_test], n_train_test= [n_train, n_test],
	# 	train_test_directories=['/Users/talhindi/Documents/claim_detection_wm/claim_features/train/',
 	#							'/Users/talhindi/Documents/claim_detection_wm/claim_features_wm/'])



	print('testing on WM narrtive data')
	n_train, n_test = 119752, 36546
	# n_train, n_test = 119752, 21852

	essays_sent_token_label, _, _, _, _ = read_wm_essays('/Users/talhindi/Documents/data_wm/arg_clean_45_2/*.tsv')

	# WM token id starts at 500000 in feature files as exported by extract_features_wm script
	token_id, sent_token_ids_test = 500000, []
	for essay in essays_sent_token_label:
		for sent_tokens, sent_labels in essay:
			sent_ids = []
			for token in sent_tokens:
				sent_ids.append(token_id)
				token_id += 1
			sent_token_ids_test.append(sent_ids)

	# combined feature extraction of sentences to test with bert embeddings
	# X_test, y_test = read_features_labels(n_test,
 #                                      sent_token_ids_test,
 #                                      '/Users/talhindi/Documents/data_wm/arg_clean_45_2/features/',
 #                                      feature_files=['lexsyn_1hop'],
 #                                      suffix='.jsonlines')
	# pickle.dump(X_test, open('wm_nr_lexsyn.p','wb'))
	# print(sent_token_ids_test)
	# print('****************\n')

	X_test, y_test = read_features_labels(n_test,
                                      sent_token_ids_test,
                                      '/Users/talhindi/Documents/data_wm/arg_clean_45_2/features/',
                                      feature_files=["structural_position", "structural_position_sent", "structural_punc", "syntactic_LCA_bin", "syntactic_POS", "syntactic_LCA_type", "lexsyn_1hop"],
                                      suffix='.jsonlines')
	pickle.dump(X_test, open('wm2_all.p','wb'))

	# run_crf_expriments(['structural_position', 'structural_punc', 'structural_position_sent','lexsyn_1hop'], 
	# 	train_test_sent_splits = [sent_token_ids_train, sent_token_ids_test], n_train_test= [n_train, n_test],
	# 	train_test_directories=['/Users/talhindi/Documents/claim_detection_wm/claim_features/train/',
	# 							'/Users/talhindi/Documents/data_wm/arg_clean_45_1/features/'])



if __name__ == '__main__':
    main()



