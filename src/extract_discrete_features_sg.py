''' August 1st, 2020
    Author: Tariq Alhindi

    Feature Extraction script for the SG2017 dataset

    script expects data directory to be entered when running the script.
    Run script as:

    python extract_features_wm.py <data_dir>

'''

import os
import json
import spacy
import pandas as pd

from utils.read_file import read_txt_and_ann
from utils.tokenize_and_label import get_labels
from utils.tokenize_and_label import get_labels_claim_premise

from features.structural_features import get_positions
from features.structural_features import get_punc_features
from features.structural_features import tok_sent_pos

from features.syntactic_features import get_pos
from features.syntactic_features import get_lca_features_doc, get_lca_features_doc_avg, get_lca_features_doc_bin
from features.syntactic_features import get_lca_types_doc

from features.lexsynproba_features import get_lex_dep_token_context
from features.lexsynproba_features import train_vectorizer, get_probability_features


class FeatureExtractor:

    def __init__(self, essay_spacy, essays_segments, train_test_split, train_feature_dir, test_feature_dir, cur_mode='merged'):
        self.essay_spacy = essay_spacy
        self.essays_segments = essays_segments
        self.train_test_split = train_test_split
        self.train_feature_dir = train_feature_dir
        self.test_feature_dir = test_feature_dir
        self.cur_mode = cur_mode


    def extract_features(self, feature_function, feature_file, hops=1):   

        open(os.path.join(self.train_feature_dir, feature_file), 'w')
        open(os.path.join(self.test_feature_dir, feature_file), 'w')

        token_id = 0
        for i, (doc, segments, group) in enumerate(zip(self.essay_spacy, self.essays_segments, self.train_test_split)):

            # some feature functions require the tokenized document and some extra parameters, such as lex_syn and probability feature functions
            if feature_function is get_lex_dep_token_context:
                features = feature_function(doc, hops)
            
            elif feature_function is get_probability_features:
                vectorizer = train_vectorizer(self.essay_spacy, self.essays_segments, self.train_test_split, get_labels)
                features = feature_function(doc, vectorizer)
            
            else:
                features = feature_function(doc)
            
            # merged means claim and permsie tokens are merge into Arg-B and Arg-I labels as done by SG2017
            if self.cur_mode == 'merged' :
                tokens, labels = get_labels(doc, segments)
            else:
                tokens, labels = get_labels_claim_premise(doc, segments, mode=self.cur_mode, labels_as_numbers=True)

            if group == "TRAIN":
                with open(os.path.join(self.train_feature_dir, feature_file), 'a') as file:
                    for f, l in zip(features, labels):
                        file.write('{{"y": {}, "x": {}, "id": {}}}\n'.format(l, json.dumps(f), token_id))
                        token_id +=1
            else:
                with open(os.path.join(self.test_feature_dir, feature_file), 'a') as file:
                    for f, l in zip(features, labels):
                        file.write('{{"y": {}, "x": {}, "id": {}}}\n'.format(l, json.dumps(f), token_id))
                        token_id +=1


def main():

    parser = argparse.ArgumentParser(description='Feature Extraction for WM data')
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the four column .tsv files with header 'sentence_id    token_id    token   label' in the first line.")
    parser.add_argument("--train_test_split_file",
                        default='',
                        type=str,
                        help="csv file provided by SG2017. It has two columns: essay_id, type. Type can be 'train' or 'test'")
    
    parser.add_argument("--arg_token_labels",
                        default=False,
                        action='store_true',
                        help="Labeling tokens as: Arg-B, Arg-I, O where Arg include both claim and premise tokens")
    parser.add_argument("--claim_token_labels",
                        default=True,
                        action='store_true',
                        help="Labeling tokens as: B-claim, I-claim, O-claim. Premise tokens are all added to the O-claim class")

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

    args = parser.parse_args()
    essays_sent_token_label, tokens, labels, essay_str, essay_str_sent = read_wm_essays(args.data_dir)
    train_test_split_file = args.train_test_split_file if len(args.train_test_split_file)>0 else os.path.join(args.data_dir, 'train-test-split.csv')

    # read files
    train_test_split = pd.read_csv(train_test_split_file, sep=';')
    _, essay_txt_str, essays_segments = read_txt_and_ann(args.data_dir, args.data_dir)


    # tokenization using spacy
    nlp = spacy.load('en_core_web_sm')

    essay_spacy = []
    for essay in essay_txt_str:
        essay_spacy.append(nlp(essay))


    if args.arg_token_labels:
        # creating folders for features of train and test data
        if not os.path.exists(os.path.join(args.data_dir, 'arg_features/')):
            os.mkdir(os.path.join(args.data_dir, 'arg_features/'))
        if not os.path.exists(os.path.join(args.data_dir, '/arg_features/train/')):
            os.mkdir(os.path.join(args.data_dir, 'arg_features/train/'))
        if not os.path.exists(os.path.join(args.data_dir, 'arg_features/test/')):
            os.mkdir(os.path.join(args.data_dir, 'arg_features/test/'))

        # instantiating an object of the FeatureExtractor class and passing the datasets
        extractor = FeatureExtractor(essay_spacy, essays_segments, train_test_split.SET,
            os.path.join(args.data_dir, 'arg_features/train/'), os.path.join(args.data_dir, 'arg_features/test/'))

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
            # extractor.extract_features(get_probability_features, 'probability.jsonlines') <-- function does not work properly



    if args.claim_token_labels:
        if not os.path.exists(os.path.join(args.data_dir, 'claim_features/')):
            os.mkdir(os.path.join(args.data_dir, 'claim_features/'))
        if not os.path.exists(os.path.join(args.data_dir, 'claim_features/train/')):
            os.mkdir(os.path.join(args.data_dir, 'claim_features/train/'))
        if not os.path.exists(os.path.join(args.data_dir, 'claim_features/test/')):
            os.mkdir(os.path.join(args.data_dir, 'claim_features/test/'))


        extractor = FeatureExtractor(essay_spacy, essays_segments, train_test_split.SET,
            os.path.join(args.data_dir, 'claim_features/train/'), os.path.join(args.data_dir, 'claim_features/test/'), cur_mode='claim')

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
            # extractor.extract_features(get_probability_features, 'probability.jsonlines') <-- function does not work properly



if __name__ == '__main__':
    main()




