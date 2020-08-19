import torch
from transformers import BertTokenizer,BertModel,BertForPreTraining,BertForQuestionAnswering
import numpy as np
import glob
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True)    


def get_individual_token_ids(sentence):
    
    T = 100
    
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    padded_tokens = tokens +['[PAD]' for _ in range(T-len(tokens))]
    attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]

    seg_ids = [1 for _ in range(len(padded_tokens))]
    sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
#     print("senetence idexes \n {} ".format(sent_ids))

    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
        
    
    return tokens, token_ids, attn_mask, seg_ids


def get_embedding(last_1_layer, last_2_layer, last_3_layer, last_4_layer):

    token_list = []
    
    for index in range(100):
        token = torch.add(last_1_layer[index],last_2_layer[index])
        token = torch.add(token,last_3_layer[index])
        token = torch.add(token,last_4_layer[index])
        #print(token)
        token_mean = torch.div(token, 4.0)
        #print(token_mean)
        token_list.append(token_mean)
        #token_mean.shape

#     print ('Shape is: %d x %d' % (len(token_list), len(token_list[0])))

#     sentence_embedding = torch.mean(torch.stack(token_list), dim=0)
#     print(sentence_embedding.shape)

    return token_list

def get_embedding_from_bert(token_ids, attn_mask, seg_ids, num_layers=4):
    bert_model.eval()

    with torch.no_grad():
        model_outputs = bert_model(token_ids, attention_mask = attn_mask,\
                                                token_type_ids = seg_ids)

    last_4_hidden_states = model_outputs[-1][-num_layers:]
#     print('**********', len(model_outputs), len(model_outputs[-1]), len(last_4_hidden_states))
#     print(token_ids)
    
    last_1_layer = torch.squeeze(last_4_hidden_states[0],dim=0)
    last_2_layer = torch.squeeze(last_4_hidden_states[1],dim=0)
    last_3_layer = torch.squeeze(last_4_hidden_states[2],dim=0)
    last_4_layer = torch.squeeze(last_4_hidden_states[3],dim=0)

    token_list_embedding = get_embedding(last_1_layer, last_2_layer, last_3_layer, last_4_layer)
    
    return token_list_embedding[:np.count_nonzero(attn_mask)]



def bert_embedding_individuals(file_name, sentences, tokenizer, bert_model, T=100):
    
    output_path = '/Users/talhindi/Documents/claim_detection/features/'
    file_name = file_name
    token_embeddings = []
    
    for i, sentence in enumerate(sentences):
        print('processing sentence: ', i)
        sent_tokens = sentence.split()
        tkns, token_ids, attn_mask, seg_ids = get_individual_token_ids(sentence)
        token_list_embedding = get_embedding_from_bert(token_ids, attn_mask, seg_ids)
        
        assert tkns[0] == '[CLS]'
        
        adjusted_token_emb, j = [], 1
        for i in range(len(sent_tokens)):
            
            # print(i+1, sent_tokens[i], end =' <--> ')
            # print(j, tkns[j], end=' ')
            j+=1
            
            if sent_tokens[i] == tkns[j-1]:
                adjusted_token_emb.append(torch.squeeze(token_list_embedding[j-1]))
            else:
                if sent_tokens[i] == tkns[j-1]+tkns[j]: # handling 's, 'm
                    adjusted_token_emb.append(torch.squeeze(torch.mean(torch.stack(token_list_embedding[j-1:j+1]))) )
                    # print(tkns[j], end=' ')
                    j+=1
                elif sent_tokens[i] == tkns[j-1]+tkns[j]+tkns[j+1]: # handling n't
                    adjusted_token_emb.append(torch.squeeze(torch.mean(torch.stack(token_list_embedding[j-1:j+2]))) )
                    # print(tkns[j], end=' ')
                    j+=2
                else:
                    wordpiece, wordpiece_emb = True, [token_list_embedding[j-1]]
                    while wordpiece:
                        if '#' in tkns[j]:
                            wordpiece_emb.append(token_list_embedding[j])
                            # print(tkns[j], end=' ')
                            j+=1
                        else:
                            wordpiece = False
                            adjusted_token_emb.append( torch.squeeze(torch.mean(torch.stack(wordpiece_emb))) )
            
            # print(j)
        
        try:
            assert tkns[j] == '[SEP]'
            assert len(sent_tokens) == len(adjusted_token_emb)
        except Exception as e: 
            np.save(os.path.join(output_path, file_name+'_'+str(i)+'.bert.npy'), token_embeddings)
            print(e)
            break
        
        token_embeddings.append(adjusted_token_emb)
        
    np.save(os.path.join(output_path, file_name+'.bert.npy'), token_embeddings)
#     np.savetxt(os.path.join(output_path,file_name+'.txt'),msg_embeddings)
    
    return token_embeddings
 


train = open('../data/SG2017_claim/train.txt','r').readlines()

sent_tokens, sentences, sent_start = [], [], 0
for i, line in enumerate(train):
    if line == '\n':
        sent = ' '.join(sent_tokens)
        sentences.append(sent)
        sent_tokens = []
    else:        
        token, label = line.rstrip().split()
        sent_tokens.append(token)

embeddings = bert_embedding_individuals('train_claim_emb', sentences[:10], tokenizer, bert_model)



# test = open('../data/SG2017_claim/test.txt','r').readlines()

# sent_tokens, sentences = [], []
# for line in test:
#     if line == '\n':
#         sent = ' '.join(sent_tokens)
#         sentences.append(sent)
#         sent_tokens = []
#     else:        
#         token, label = line.rstrip().split()
#         sent_tokens.append(token)

# embeddings = bert_embedding_individuals('test_claim_emb', sentences, tokenizer, bert_model)




# test = open('../data/SG2017_claim/test.txt','r').readlines()

# sent_tokens, sentences = [], []
# for line in test:
#     if line == '\n':
#         sent = ' '.join(sent_tokens)
#         sentences.append(sent)
#         sent_tokens = []
#     else:        
#         token, label = line.rstrip().split()
#         sent_tokens.append(token)

# embeddings = bert_embedding_individuals('test_claim_emb', sentences, tokenizer, bert_model)























