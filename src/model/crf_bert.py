def merge_features(bert_features, other_features):
    
    for sent_emb_features, sent_other_features in zip(bert_features, other_features):
        
        for word_emb_features, word_other_features in zip(sent_emb_features[:len(sent_other_features)], sent_other_features):
            word_other_features.update(word_emb_features)
        
        if len(sent_other_features) > len(sent_emb_features):
            for _ in range(len(sent_other_features)-len(sent_emb_features)):
                sent_other_features.pop()