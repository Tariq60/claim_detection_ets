''' August 1st, 2020
    Author: Tariq Alhindi

    Feature Extraction script for the writting mentor data

    script expects data directory to be entered when running the script.
    Run script as:

    python extract_features_wm.py <data_dir>

'''

import os
import json
import spacy
import pandas as pd
import glob
from sys import argv

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
                starting_token_id = 500000,
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
            
            # some feature functions require the tokenized document and some extra parameters, such as lex_syn and probability feature functions
            if feature_function is get_lex_dep_token_context:
                features = feature_function(doc, hops)
            
            elif feature_function is get_probability_features:
                vectorizer = train_vectorizer(self.essay_spacy, self.essays_segments, self.train_test_split, get_labels)
                features = feature_function(doc, vectorizer)
            
            else: # otherwise we just pass the tokenized document
                features = feature_function(doc)

            # print(len(features), len(doc), len(doc_tokens), len(doc_labels))
            assert len(features) == len(doc) == len(doc_tokens) == len(doc_labels)

            with open(os.path.join(self.feature_dir, feature_file), 'a') as file:
                for f, l, spacy_token, token in zip(features, doc_labels, doc, doc_tokens):
                    assert token == spacy_token.string.strip()
                    file.write('{{"y": {}, "x": {}, "id": {}}}\n'.format(self.label_dict[l], json.dumps(f), token_id))
                    token_id +=1
                    print(token, token_id)
            

def main():


    if len(sys.argv) < 1:
        print('Missing data directory. Please run the script as the following: \
               python extract_features_wm.py <data_dir>')
        sys.exit()
    elif len(sys.argv) > 1:
        print('Too many arguments. Please run the script as the following: \
               python extract_features_wm.py <data_dir>')
        sys.exit()
    else:
        data_dir = sys.argv[1]
    

    # data_dir = '/Users/talhindi/Documents/data_wm/arg_clean_45_1/*.tsv'
    essays_sent_token_label, tokens, labels, essay_str, essay_str_sent = read_wm_essays(data_dir)

    # tokenization using spacy
    nlp = spacy.load('en_core_web_sm')

    essay_spacy = []
    for essay in essay_str:
        essay_spacy.append(nlp(essay))


    # creating folders for features of train and test data
    if not os.path.exists(os.path.join(data_dir, '/features/')):
        os.mkdir(os.path.join(data_dir, '/features/'))

    print('Features will be exported to: ',os.path.join(data_dir, '/features/'))

    # instantiating an object of the FeatureExtractorWM class and passing the datasets
    extractor = FeatureExtractorWM(essay_spacy, tokens, labels, open(os.path.join(data_dir, '/features/'))

    #  extract structural features
    extractor.extract_features(get_positions, 'structural_position_test.jsonlines')
    extractor.extract_features(get_punc_features, 'structural_punc.jsonlines')
    extractor.extract_features(tok_sent_pos, 'structural_position_sent.jsonlines')


    # #  extract syntactic features
    extractor.extract_features(get_pos, 'syntactic_POS.jsonlines')
    extractor.extract_features(get_lca_features_doc, 'syntactic_LCA.jsonlines')
    extractor.extract_features(get_lca_features_doc_avg, 'syntactic_LCA_avg.jsonlines')
    extractor.extract_features(get_lca_features_doc_bin, 'syntactic_LCA_bin.jsonlines')
    extractor.extract_features(get_lca_types_doc, 'syntactic_LCA_type.jsonlines')


    # #  extract LexSyn and Probability features
    extractor.extract_features(get_lex_dep_token_context, 'lexsyn_1hop.jsonlines')
    extractor.extract_features(get_lex_dep_token_context, 'lexsyn_2hops.jsonlines', hops=2)
    # extractor.extract_features('probability.jsonlines')  <-- function does not work properly



if __name__ == '__main__':
    main()




