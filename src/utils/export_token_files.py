'''
This is a backed-up script that might not be needed.
If needed, you can modify the paths and pass the path to the SG2017 to export the four column .tsv files as we use in the WM2020 dataset
'''



# claim + premise merged
train_file  = open('../data/SG2017_tok/train.txt', 'w')
test_file  = open('../data/SG2017_tok/test.txt', 'w')

# train_dep_file  = open('../data/SG2017_tokenized/dep/train.txt', 'w')
# test_dep_file  = open('../data/SG2017_tokenized/dep/test.txt', 'w')

for essay_id, (doc, segments, group) in enumerate(zip(essay_spacy, essays_segments, train_test_split.SET)):
    
    tokens, labels = get_labels(doc, segments)
    labeled_token_id = 0
    
    if essay_id+1 < 10:
        essay_3digit_id = '00'+str(essay_id+1)
    elif essay_id+1 < 100:
        essay_3digit_id = '0'+str(essay_id+1)
    else:
        essay_3digit_id = str(essay_id+1)
    
    if group == "TRAIN":
        with open('../data/SG2017_tok/train/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        train_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
#                         train_dep_file.write('{}_{} {}\n'.format(token.text, token.dep_, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                train_file.write('\n')
#                 train_dep_file.write('\n')
                
    else:
        with open('../data/SG2017_tok/test/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        test_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
#                         test_dep_file.write('{}_{} {}\n'.format(token.text, token.dep_, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                test_file.write('\n')  
#                 test_dep_file.write('\n')
                




# claim + premise merged with dep relation
train_dep_file  = open('../data/SG2017_tok_dep/train.txt', 'w')
test_dep_file  = open('../data/SG2017_tok_dep/test.txt', 'w')

for essay_id, (doc, segments, group) in enumerate(zip(essay_spacy, essays_segments, train_test_split.SET)):
    
    tokens, labels = get_labels(doc, segments)
    labeled_token_id = 0
    
    if essay_id+1 < 10:
        essay_3digit_id = '00'+str(essay_id+1)
    elif essay_id+1 < 100:
        essay_3digit_id = '0'+str(essay_id+1)
    else:
        essay_3digit_id = str(essay_id+1)
    
    if group == "TRAIN":
        with open('../data/SG2017_tok_dep/train/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        train_dep_file.write('{}_{} {}\n'.format(token.text, token.dep_, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                train_dep_file.write('\n')
                
    else:
        with open('../data/SG2017_tok_dep/test/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        test_dep_file.write('{}_{} {}\n'.format(token.text, token.dep_, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                test_dep_file.write('\n')
                



# claim only
train_file  = open('../data/SG2017_claim/train.txt', 'w')
test_file  = open('../data/SG2017_claim/test.txt', 'w')


for essay_id, (doc, segments, group) in enumerate(zip(essay_spacy, essays_segments, train_test_split.SET)):
    
    tokens, labels = get_labels_claim_premise(doc, segments, mode='claim')
    labeled_token_id = 0
    
    if essay_id+1 < 10:
        essay_3digit_id = '00'+str(essay_id+1)
    elif essay_id+1 < 100:
        essay_3digit_id = '0'+str(essay_id+1)
    else:
        essay_3digit_id = str(essay_id+1)
    
    if group == "TRAIN":
        with open('../data/SG2017_claim/train/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        train_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                train_file.write('\n')
                
    else:
        with open('../data/SG2017_claim/test/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        test_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                test_file.write('\n')  
                



# premise only
train_file  = open('../data/SG2017_premise/train.txt', 'w')
test_file  = open('../data/SG2017_premise/test.txt', 'w')


for essay_id, (doc, segments, group) in enumerate(zip(essay_spacy, essays_segments, train_test_split.SET)):
    
    tokens, labels = get_labels_claim_premise(doc, segments, mode='premise')
    labeled_token_id = 0
    
    if essay_id+1 < 10:
        essay_3digit_id = '00'+str(essay_id+1)
    elif essay_id+1 < 100:
        essay_3digit_id = '0'+str(essay_id+1)
    else:
        essay_3digit_id = str(essay_id+1)
    
    if group == "TRAIN":
        with open('../data/SG2017_premise/train/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        train_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                train_file.write('\n')
                
    else:
        with open('../data/SG2017_premise/test/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        test_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                test_file.write('\n')  
                





# claim and premise separated
train_file  = open('../data/SG2017_claim_premise/train.txt', 'w')
test_file  = open('../data/SG2017_claim_premise/test.txt', 'w')


for essay_id, (doc, segments, group) in enumerate(zip(essay_spacy, essays_segments, train_test_split.SET)):
    
    tokens, labels = get_labels_claim_premise(doc, segments, mode='all')
    labeled_token_id = 0
    
    if essay_id+1 < 10:
        essay_3digit_id = '00'+str(essay_id+1)
    elif essay_id+1 < 100:
        essay_3digit_id = '0'+str(essay_id+1)
    else:
        essay_3digit_id = str(essay_id+1)
    
    if group == "TRAIN":
        with open('../data/SG2017_claim_premise/train/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        train_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                train_file.write('\n')
                
    else:
        with open('../data/SG2017_claim_premise/test/essay{}.tsv'.format(essay_3digit_id), 'w') as file:
            file.write('sentence_id\ttoken_id\ttoken\tlabel\n')
            for sent_id, sent in enumerate(doc.sents):
                for token_id, token in enumerate(sent):
                    assert token.text == tokens[labeled_token_id]
                    file.write('{}\t{}\t{}\t{}\n'.format(sent_id, token_id, token.text.replace('\n','_NEW_LINE_'), labels[labeled_token_id]))
                    
                    if '\n' not in token.text:
                        test_file.write('{} {}\n'.format(token.text, labels[labeled_token_id]))
                    
                    labeled_token_id += 1
                    
                test_file.write('\n')  
                