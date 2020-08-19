from itertools import chain
import os
import copy
import json
import nltk
import glob
import pandas as pd
import spacy

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
# from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics




def read_features_labels(n_examples, sent_token_ids, directory, feature_files, suffix):
    features, labels, feature_sets = [], [], []
    
    for file in feature_files:
        feature_sets.append(open(directory + file + suffix, 'r').readlines())
    
    for fset in feature_sets:
        assert len(fset) == n_examples
    
    sent_id, token_id = 0, 0
    sent, sent_labels = [], []
    for i in range(n_examples):
        
        token_features = {}
        for fset in feature_sets:
            jsonline = json.loads(fset[i])
            for key in jsonline['x'].keys():
                if type(jsonline['x'][key]) is int:
                    token_features[key] = float(jsonline['x'][key])
                else:
                    token_features[key] = jsonline['x'][key]
        
        # print(sent_id, token_id, sent_token_ids[sent_id][token_id], jsonline['id'])
        assert sent_token_ids[sent_id][token_id] == jsonline['id']
        
        sent.append(copy.deepcopy(token_features))
        sent_labels.append(str(jsonline['y']))
        del token_features
        
        if token_id == len(sent_token_ids[sent_id])-1:
            # print(jsonline['id'], len(sent), len(sent_labels), len(sent_token_ids[sent_id]))
            assert len(sent) == len(sent_labels) == len(sent_token_ids[sent_id])
            
            features.append(sent)
            labels.append(sent_labels)
            
            sent, sent_labels = [], []
            sent_id += 1; token_id = 0
        else:
            token_id += 1
    
    return features, labels



# n_train, n_test = 119752, 29815

def run_crf_expriments(features, train_test_sent_splits, n_train_test, suffix='.jsonlines',
                   train_test_directories = ['../features/SG2017_train/', '../features/SG2017_test/'],
                   ):
    'Takes a '

    X_train, y_train = read_features_labels(n_train_test[0], train_test_sent_splits[0],
                                            train_test_directories[0], features, suffix)
    X_test, y_test = read_features_labels(n_train_test[1], train_test_sent_splits[1],
                                            train_test_directories[1], features, suffix)
    
    print('train dir: ', train_test_directories[0])
    print('test dir: ', train_test_directories[1])
    print('\nReading the following features: ')
    print(features)
    
    print('\nBuilding  CRF model')
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    print(crf)
    
    print('\nTraining Model...')
    crf.fit(X_train, y_train)
    
    print('\nPredictions on the test set')
    y_pred = crf.predict(X_test)
    print('Macro F1: ', metrics.flat_f1_score(y_test, y_pred, average='macro', labels=['0.0','1.0','2.0']))
    
    print('\nclassification report:')
    y_test_flat = [y for y_seq in y_test for y in y_seq]
    y_pred_flat = [y for y_seq in y_pred for y in y_seq]
    print(classification_report(y_test_flat, y_pred_flat, digits=3))



    