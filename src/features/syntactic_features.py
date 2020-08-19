import networkx as nx
import numpy as np

'''1. Part-of-speech: The token’s part-of-speech'''
def get_pos(essay_spacy):
    
    pos_features = []
    for token in essay_spacy:
        pos_features.append({'pos_{}'.format(token.pos_): 1.0})
    
    return pos_features
            


'''2. Lowest common ancestor (LCA): Normalized length of the path to the LCA with the *following* and *preceding* token in the parse tree'''
def get_lca_features_doc(doc):
    token_lca, sent_id = [], 0
    
    for sent in doc.sents:
#         print(sent_id)
        if len(sent) > 1:
            sent_lca = _get_lca_features_sent(sent, False, False, False)
        else:
            assert len(sent) == 1
            sent_lca = [{'lca_prev_path': 0, 'lca_next_path': 0}]
        
        for feature in sent_lca:
            token_lca.append(feature)
        sent_id += 1
    
    return token_lca

'''2. Lowest common ancestor (LCA): Normalized *average* length of the path to the LCA with the *following* and *preceding* token in the parse tree'''
def get_lca_features_doc_avg(doc, get_average=True):
    token_lca, sent_id = [], 0
    
    for sent in doc.sents:
#         print(sent_id)
        if len(sent) > 1:
            sent_lca = _get_lca_features_sent(sent, get_average, False, False)
        else:
            assert len(sent) == 1
            sent_lca = [{'lca_prev_path': 0, 'lca_next_path': 0}]
        
        for feature in sent_lca:
            token_lca.append(feature)
        sent_id += 1
    
    return token_lca

'''2. Lowest common ancestor (LCA) binarized'''
def get_lca_features_doc_bin(doc, binarize=True):
    token_lca, sent_id = [], 0
    
    for sent in doc.sents:
#         print(sent_id)
        if len(sent) > 1:
            sent_lca = _get_lca_features_sent(sent, False, False, binarize)
        else:
            assert len(sent) == 1
            sent_lca = [{'lca_prev_path': 0, 'lca_next_path': 0}]
        
        for feature in sent_lca:
            token_lca.append(feature)
        sent_id += 1
    
    return token_lca



'''3. LCA types: The two constituent types of the LCA of the current token and its preceding and following token'''
def get_lca_types_doc(doc):
    
    token_lca, sent_id = [], 0
    
    for sent in doc.sents:
#         print(sent_id)
        if len(sent) > 1:
            sent_lca = _get_lca_features_sent(sent, False, True, False)
        else:
            assert len(sent) == 1
            sent_lca = [{}]
        
        for feature in sent_lca:
            token_lca.append(feature)
        sent_id += 1
    
    return token_lca
    



# Getting LCA features for each sentence    
def _get_lca_features_sent(sent, get_average, get_types=False, binarize=False):
    
    edges = []
    for token in sent:
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.text, token.i),'{0}-{1}'.format(child.text, child.i)))
            
    graph = nx.Graph(edges)
    lca_matrix = sent.get_lca_matrix()
#     print(graph, edges, lca_matrix)
    
    lca_prev_next_path_sent, lca_types_sent = [], []
    for token_id, token in enumerate(sent):
        if token_id == 0:
            token_prev_lca = -1
            token_next_lca = lca_matrix[token_id, token_id+1]
        elif token_id == len(sent)-1:
            token_prev_lca = lca_matrix[token_id, token_id-1]
            token_next_lca = -1
        else:
            token_prev_lca = lca_matrix[token_id, token_id-1]
            token_next_lca = lca_matrix[token_id, token_id+1]
        
        # adding index to tokens to retrieve node in graph. node-name = token-index
        source_token = '{0}-{1}'.format(token.text, token.i)
        lca_types_token = {}
        
        # token, previous_token shortest path to lca
        if token_prev_lca != -1:
            source_prev_token = '{0}-{1}'.format(sent[token_id-1].text, sent[token_id-1].i)
            target_token_prev_lca = '{0}-{1}'.format(sent[token_prev_lca].text, sent[token_prev_lca].i)
#             lca_types_token['lca_prev_{}'.format(sent[token_prev_lca].pos_)] = 1.0
            lca_types_token['lca_prev_{}'.format(sent[token_prev_lca].dep_)] = 1.0
            
            lca_prev_path_token = nx.shortest_path_length(graph, source=source_token, target=target_token_prev_lca)
            lca_prev_path_prev = nx.shortest_path_length(graph, source=source_prev_token, target=target_token_prev_lca)
            
            if get_average:
                lca_prev_path = np.mean((lca_prev_path_token, lca_prev_path_prev))
            else:
                lca_prev_path = lca_prev_path_token
        else:
            lca_prev_path = -1
            
        # token, next_token shortest path to lca
        if token_next_lca != -1:
            source_next_token = '{0}-{1}'.format(sent[token_id+1].text, sent[token_id+1].i)
            target_token_next_lca = '{0}-{1}'.format(sent[token_next_lca].text, sent[token_next_lca].i)
#             lca_types_token['lca_next_{}'.format(sent[token_next_lca].pos_)] = 1.0
            lca_types_token['lca_next_{}'.format(sent[token_next_lca].dep_)] = 1.0
            
            lca_next_path_token = nx.shortest_path_length(graph, source=source_token, target=target_token_next_lca)
            lca_next_path_next = nx.shortest_path_length(graph, source=source_next_token, target=target_token_next_lca)
            
            if get_average:
                lca_next_path = np.mean((lca_next_path_token, lca_next_path_next))
            else:
                lca_next_path = lca_next_path_token
        else:
            lca_next_path = -1
        
        # adding LCA features of this token
        if get_average:
            lca_prev_next_path_sent.append({'lca_prev_path_avg': lca_prev_path, 'lca_next_path_avg': lca_next_path})
        elif binarize:
            lca_prev_next_path_sent.append({'lca_prev_path_{}'.format(lca_prev_path): 1.0,
                                            'lca_next_path_{}'.format(lca_next_path): 1.0})
        else:
            lca_prev_next_path_sent.append({'lca_prev_path': lca_prev_path, 'lca_next_path': lca_next_path})
        
        # adding LCA-type feature of this token
        lca_types_sent.append(lca_types_token)
     
    # returning LCA features for all tokens in the sentence
    if not get_types:
        return lca_prev_next_path_sent
    else:
        return lca_types_sent
    