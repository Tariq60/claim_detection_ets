import copy
from sklearn.feature_extraction.text import CountVectorizer

'''SG2017-LexSyn 1:
        We use lexical head projection rules (Collins 2003) implemented in the Stanford tool suite
        to lexicalize the constituent parse tree. 
        For each token t, we extract its uppermost node n in the parse tree 
        with the lexical head t and define a lexico- syntactic feature as 
        the combination of t and the constituent type of n.'''

def get_lex_dep_token_context(doc, hops=1):
    '''A modification of SG2017 LexSyn features. We get the relation governing the token and its previous and
    next tokens. We also go N hops deep in retrieving those relations where N(hops) is a input to this function'''
    features = []
    for sent in doc.sents:
        for i, token in enumerate(sent):
            token_features = {}
            this_token = token
            prev_token = sent[i-1] if i > 0 else 'NO_TOKEN' 
            next_token = sent[i+1] if i < len(sent)-1 else 'NO_TOKEN'
            
#             print(this_token, prev_token, next_token)
            
            for j in range(hops):
#                 print(j)
                if type(this_token) is not str:
                    token_features['dep_{}_{}'.format(j, this_token.dep_)] = 1.0
                    token_features['token_dep_{}_{}_{}'.format(j, this_token.dep_, this_token)] = 1.0
                    this_token = token.head if this_token.dep_ != 'ROOT' else 'NO_TOKEN'
                
                if type(prev_token) is not str:
                    _get_lex_dep_token_prev(sent[i-1], token_features, j)
                    prev_token = token.head if prev_token.dep_ != 'ROOT' else 'NO_TOKEN'
                
                if type(next_token) is not str:
                    _get_lex_dep_token_next(sent[i+1], token_features, j)
                    next_token = token.head if next_token.dep_ != 'ROOT' else 'NO_TOKEN'

            features.append(copy.deepcopy(token_features))
            del token_features
    return features

def _get_lex_dep_token_prev(prev_token, token_features, j):
    token_features['prev_dep_{}_{}'.format(j, prev_token.dep_)] = 1.0
    token_features['prev_token_dep_{}_{}_{}'.format(j,prev_token.dep_, prev_token)] = 1.0

def _get_lex_dep_token_next(next_token, token_features, j):
    token_features['next_dep_{}_{}'.format(j, next_token.dep_)] = 1.0
    token_features['next_token_dep_{}_{}_{}'.format(j, next_token.dep_, next_token)] = 1.0
            


'''SG2017-LexSyn 2:
        We also consider the child node of n in the path to t and its right sibling, 
        and combine their lexical heads and constituent types as described by Soricut and Marcu (2003).'''
def get_sibling_features(doc):
    pass



'''Probability Feature
        the conditional probability of the current token t_i being the beginning of an argument component (“Arg-B”)
        given its preceding tokens (up to 3 prev_tokens) using MLE on the training data
'''
def train_vectorizer(essay_spacy, essays_segments, train_test_split, labeling_function, B_labels='Arg-B'):

    argB_train_segments = []
    for essay, segments, group in zip(essay_spacy, essays_segments, train_test_split):
        tokens, labels = labeling_function(essay, segments, labels_as_numbers=False)

        for i, (t, l)  in enumerate(zip(tokens, labels)):
            if l == B_labels and group == 'TRAIN':
            	argB_train_segments.append(' '.join([tokens[i-3],tokens[i-2],tokens[i-1]]) )
        
    vect = CountVectorizer(ngram_range=(1,3))
    vect.fit(argB_train_segments)
        
    return vect
        
        

def get_probability_features(doc, vectorizer):
    
    features = []
    for i, token in enumerate(doc):
        if i == 0:
            prev_context = ''
        elif i == 1:
            prev_context = doc[0].text
        elif i == 2:
            prev_context = ' '.join([doc[0].text, doc[1].text])
        else:
            prev_context = ' '.join([doc[i-3].text, doc[i-2].text, doc[i-1].text])
            
        grams = vectorizer.transform([prev_context])[0]
        features.append({'probability_feature': grams.count_nonzero()/ grams.shape[1]})
    
    return features

