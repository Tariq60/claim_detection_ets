''' August 28th, 2020
    Author: Tariq Alhindi

    Feature Extraction script for the writting mentor data (or anything in the same format)
'''

import os
import json
import spacy
import pandas as pd
import glob
import argparse
import timeit

from utils.read_file import read_wm_essays

from features.structural_features import get_positions
from features.structural_features import get_punc_features
from features.structural_features import tok_sent_pos

from features.syntactic_features import get_pos
from features.syntactic_features import get_lca_features_doc, get_lca_features_doc_avg, get_lca_features_doc_bin
from features.syntactic_features import get_lca_types_doc

from features.lexsynproba_features import get_lex_dep_token_context
from features.lexsynproba_features import train_vectorizer, get_probability_features


class FeatureExtractorWM:


    def __init__(self, essay_spacy, tokens, labels, feature_dir,
                starting_token_id = 0,
                label_dict={'O-claim': 0.0, 'B-claim': 1.0, 'I-claim': 2.0}
                ):

        self.essay_spacy = essay_spacy
        self.tokens = tokens
        self.labels = labels
        self.feature_dir = feature_dir
        self.starting_token_id = starting_token_id
        self.label_dict = label_dict

    
    def extract_features(self, feature_function, feature_file, hops=1):   

        open(os.path.join(self.feature_dir, feature_file), 'w')

        token_id = self.starting_token_id
        for i, (doc, doc_tokens, doc_labels) in enumerate(zip(self.essay_spacy, self.tokens, self.labels)):
            
            # print(repr(doc.text))
            # some feature functions require the tokenized document and some extra parameters, such as lex_syn and probability feature functions
            if feature_function is get_lex_dep_token_context:
                features = feature_function(doc, hops)
            
            elif feature_function is get_probability_features:
                vectorizer = train_vectorizer(self.essay_spacy, self.essays_segments, self.train_test_split, get_labels)
                features = feature_function(doc, vectorizer)
            
            else: # otherwise we just pass the tokenized document
                features = feature_function(doc)

            print(i, len(features), len(doc), len(doc_tokens), len(doc_labels))
            # assert len(features) == len(doc) == len(doc_tokens) == len(doc_labels)

            with open(os.path.join(self.feature_dir, feature_file), 'a') as file:
                for f, l, spacy_token, token in zip(features, doc_labels, doc, doc_tokens):
                    if token != spacy_token.string.strip():
                        print(token, spacy_token.string.strip())
                    # assert token == spacy_token.string.strip()
                    file.write('{{"y": {}, "x": {}, "id": {}}}\n'.format(self.label_dict[l], json.dumps(f), token_id))
                    token_id +=1
                    # print(token, token_id)
            

def main():

    parser = argparse.ArgumentParser(description='Feature Extraction for WM data')
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the four column .tsv files with header 'sentence_id    token_id    token   label' in the first line.")
    parser.add_argument("--extract_structural",
                        default=True,
                        action='store_true',
                        help="Extracting structural featutres when set to True.")
    parser.add_argument("--extract_syntactic",
                        default=True,
                        action='store_true',
                        help="Extracting syntactic featutres when set to True.")
    parser.add_argument("--extract_lexsyn",
                        default=True,
                        action='store_true',
                        help="Extracting lexico-syntactic featutres when set to True.")
    parser.add_argument("--start_token_id",
                        default=0,
                        type=int,
                        help="starting token index id in the exported feature files.")

    args = parser.parse_args()
    essays_sent_token_label, tokens, labels, essay_str, essay_str_sent = read_wm_essays(args.data_dir)
    label_set = set([item for _list in labels for item in _list])
    label_dict, v = {}, 0.0
    for k in sorted(label_set):
        label_dict[k] = v
        v += 1
    print('labels are:', label_dict)

    print('now tokenizing essays using spacy...')
    # tokenization using spacy
    nlp = spacy.load('en_core_web_sm')

    start = timeit.default_timer()
    essay_spacy = []
    for essay in essay_str:
        essay_spacy.append(nlp(essay.strip()))
    stop = timeit.default_timer()

    print('tokenization took {} for {} essays'.format(stop-start, len(essay_spacy)))

    # creating folders for features of train and test data
    if not os.path.exists(os.path.join(args.data_dir, 'features/')):
        os.makedirs(os.path.join(args.data_dir, 'features/'))

    print('Features will be exported to: ',os.path.join(args.data_dir, 'features/'))

    # instantiating an object of the FeatureExtractorWM class and passing the datasets
    extractor = FeatureExtractorWM(essay_spacy, tokens, labels, os.path.join(args.data_dir, 'features/'), args.start_token_id, label_dict)

    start = timeit.default_timer()
    #  extract structural features
    if args.extract_structural:
        extractor.extract_features(get_positions, 'structural_position.jsonlines')
        extractor.extract_features(get_punc_features, 'structural_punc.jsonlines')
        extractor.extract_features(tok_sent_pos, 'structural_position_sent.jsonlines')


    #  extract syntactic features
    if args.extract_syntactic:
        extractor.extract_features(get_pos, 'syntactic_POS.jsonlines')
        extractor.extract_features(get_lca_features_doc, 'syntactic_LCA.jsonlines')
        extractor.extract_features(get_lca_features_doc_avg, 'syntactic_LCA_avg.jsonlines')
        extractor.extract_features(get_lca_features_doc_bin, 'syntactic_LCA_bin.jsonlines')
        extractor.extract_features(get_lca_types_doc, 'syntactic_LCA_type.jsonlines')


    #  extract LexSyn and Probability features
    if args.extract_lexsyn:
        extractor.extract_features(get_lex_dep_token_context, 'lexsyn_1hop.jsonlines')
        extractor.extract_features(get_lex_dep_token_context, 'lexsyn_2hops.jsonlines', hops=2)
    # extractor.extract_features('probability.jsonlines')  <-- function does not work properly
    stop = timeit.default_timer()

    print('feature extraction took {} for {} essays'.format(stop-start, len(essay_spacy)))

if __name__ == '__main__':
    main()




