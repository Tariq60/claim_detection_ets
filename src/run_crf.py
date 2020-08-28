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

	# Directories
 	train_data_dir = '/Users/talhindi/Documents/data_wm/arg_clean_45_2/'
	train_feature_dir = train_data_dir + 'features/'

 	test_data_dir = '/Users/talhindi/Documents/data_wm/arg_clean_45_1/'
	test_feature_dir = test_data_dir + 'features/'

	features_list = ["structural_position", "structural_position_sent", "structural_punc", "syntactic_LCA_bin", "syntactic_POS", "syntactic_LCA_type", "lexsyn_1hop"]



	print('training and testing on WM narrtive data')
	n_train, n_test = 36546, 21852 # total tokens in each file
	

	# prepraing train features in sequence formats because extract_features.py outputs token features only
	train_essays_sent_token_label, _, _, _, _ = read_wm_essays(train_data_dir)

	# WM token id starts at 500000 in feature files as exported by extract_features_wm script
	token_id, sent_token_ids_train = 500000, []
	for essay in train_essays_sent_token_label:
		for sent_tokens, sent_labels in essay:
			sent_ids = []
			for token in sent_tokens:
				sent_ids.append(token_id)
				token_id += 1
			sent_token_ids_test.append(sent_ids)

	X_train, y_train = read_features_labels(n_train,
                                      sent_token_ids_train,
                                      train_feature_dir,
                                      feature_files=features_list,
                                      suffix='.jsonlines')


	# prepraing test features in sequence formats because extract_features.py outputs token features only
	test_essays_sent_token_label, _, _, _, _ = read_wm_essays(test_data_dir)

	# WM token id starts at 500000 in feature files as exported by extract_features_wm script
	token_id, sent_token_ids_test = 500000, []
	for essay in test_essays_sent_token_label:
		for sent_tokens, sent_labels in essay:
			sent_ids = []
			for token in sent_tokens:
				sent_ids.append(token_id)
				token_id += 1
			sent_token_ids_test.append(sent_ids)

	X_test, y_test = read_features_labels(n_test,
                                      sent_token_ids_test,
                                      test_feature_dir,
                                      feature_files=features_list,
                                      suffix='.jsonlines')
	
	# pickle.dump(X_test, open('wm2_all.p','wb'))

	
	# training crf model
	run_crf_expriments(features_list, 
						train_test_sent_splits = [sent_token_ids_train, sent_token_ids_test],
						n_train_test= [n_train, n_test],
						train_test_directories= [train_feature_dir, test_feature_dir]
					   )



if __name__ == '__main__':
    main()



