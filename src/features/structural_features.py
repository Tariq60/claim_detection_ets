import copy

'''1. token position features: 
    Token present in introduction or conclusion; 
    Token is first or last token in sentence; 
    Relative and absolute token position in document, paragraph and sentence'''

def get_positions(essay):
    
    doc_len, prg_lengths, sent_lengths = _get_lengths(essay)
    
    positions = []
    doc_pos = 0; prg_id = 0; prg_pos = 0
    for sent_id, sent in enumerate(essay.sents):
        sent_pos = 0
        if len(sent)> 1:
            last_word_index = [i for i, token in enumerate(sent) if not token.is_punct and not token.is_space][-1]
        else:
            last_word_index = 0
            
        for i, token in enumerate(sent):
            token_features = {}

            if i == 0: 
                token_features['is_first_in_sent'] = 1.0
            elif i == last_word_index: 
                token_features['is_last_in_sent'] = 1.0

            if prg_id == 1: # 0 is title
                token_features['in_introduction'] = 1.0
            elif prg_id == len(prg_lengths)-1:
                token_features['in_conclusion'] = 1.0

            token_features['doc_abs_pos'] = doc_pos
            token_features['prg_abs_pos'] = prg_pos
            token_features['sent_abs_pos'] = sent_pos
            token_features['doc_rel_pos'] = round(doc_pos/doc_len,4)
            token_features['prg_rel_pos'] = round(prg_pos/prg_lengths[prg_id],4)
            token_features['sent_rel_pos'] = round(sent_pos/sent_lengths[sent_id],4)

            positions.append(copy.deepcopy(token_features))            
            doc_pos += 1; prg_pos += 1; sent_pos += 1;  
            del token_features
        
        # finding paragraphs cuts
        if token.string in ['\n','\n\n']:
            prg_id += 1
            prg_pos = 0
    
    return positions



def _get_lengths(essay):
    '''Returns essay length, and length of each paragraph and sentence in the essay'''
    
    prg_lengths, sent_lengths = [], []
    doc_len, prg_len, end_prg = 0, 0, False
    
    for sent in essay.sents:
        
        if end_prg:
            prg_lengths.append(prg_len)
            prg_len = 0; end_prg = False
        
        sent_len = 0
        for token in sent:
            doc_len += 1; prg_len += 1; sent_len += 1
            if token.string in ['\n','\n\n']:
                end_prg = True
        
        sent_lengths.append(sent_len)
    
    prg_lengths.append(prg_len)
    
    assert doc_len == sum(prg_lengths) == sum(sent_lengths)
    return doc_len, prg_lengths, sent_lengths



#------------------------------------------------------------------------

'''2. punctuation features:
    Token precedes or follows any punctuation, full stop, comma and semicolon;
    Token is any punctuation or full stop'''      


def get_punc_features(essay):
    
    token_features = {}
    _set_reset_features(token_features)
    
    features = []
    for i, token in enumerate(essay):
#         print(i, token)
        if token.is_punct:
            token_features['punc'] = True
            if token.text == ".": token_features['fullstop'] = True
                    
        if i == 0:
            _next_punc_features(essay[i+1], token_features)
        elif i == len(essay)-1:
            _prev_punc_features(essay[i-1], token_features)
        else:
            _prev_punc_features(essay[i-1], token_features)
            _next_punc_features(essay[i+1], token_features)

        # adding 'true' features of this token to the list of features
        token_punc_features  = _token_punc_features(token_features)
        features.append(copy.deepcopy(token_punc_features))

        # resetting features to process the next token
        _set_reset_features(token_features)
        
    return features

    
def _set_reset_features(token_features):
    token_features['punc'], token_features['fullstop'] = False, False
    token_features['punc_prev'], token_features['fullstop_prev'] = False, False
    token_features['comma_prev'], token_features['semicolon_prev'] = False, False
    token_features['punc_next'], token_features['fullstop_next'] = False, False
    token_features['comma_next'], token_features['semicolon_next'] = False, False

def _prev_punc_features(prev_token, token_features):    
    if prev_token.is_punct:
        token_features['punc_prev'] = True
        if prev_token.text == ".":
            token_features['fullstop_prev'] = True
        if prev_token.text == ",":
            token_features['comma_prev'] = True
        if prev_token.text == ";":
            token_features['semicolon_prev'] = True

def _next_punc_features(next_token, token_features):
    if next_token.is_punct:
        token_features['punc_next'] = True
        if next_token.text == ".":
            token_features['fullstop_next'] = True
        if next_token.text == ",":
            token_features['comma_next'] = True
        if next_token.text == ";":
            token_features['semicolon_next'] = True

def _token_punc_features(this_token_features):
    features = {}
    
    for key in sorted(this_token_features.keys()):
        value = this_token_features[key]
        if value:
            features[key] = 1.0
        
    return features




#------------------------------------------------------------------------

'''3. position of covering sentence
    Absolute and relative position of the token’s covering sentence in the document and paragraph'''

def tok_sent_pos(essay):
    
    doc_len, prg_lengths, _ = _get_lengths(essay)
    
    doc_pos, prg_pos, prg_id, positions = 0, 0, 0, []
    for sent in essay.sents:
        for token in sent:
            positions.append({'sent_doc_abs_pos': doc_pos, 'sent_prg_abs_pos': prg_pos,
                          'sent_doc_rel_pos': round(doc_pos/doc_len,4), 'sent_prg_rel_pos': round(prg_pos/prg_lengths[prg_id],4)})
        doc_pos += 1; prg_pos += 1;
        
        if token.string in ['\n','\n\n']:
            prg_id += 1
            prg_pos = 0

    return positions