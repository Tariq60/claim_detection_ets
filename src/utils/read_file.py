import glob
import os


def read_txt_and_ann(txt_dir,   # directory to the text files of the essays
                     ann_dir,   # directory to the annotated files of the essays
                    ):
    ''' Returns three lists, essays as lists of paragraphs, essays as strings, and a list of essays labeled segments'''

    # readin the text files of the essays, where each essays is a list of paragraphs
    essays_txt_prg_list = []
    for file in sorted(glob.glob( os.path.join(txt_dir,"*.txt") )):
        essay = open(file).readlines()
        essays_txt_prg_list.append(essay)

    # merging the paragraphs of each essay in one string for the whole essay
    essay_txt_str = []
    for essay in essays_txt_prg_list:
        essay_txt_str.append(''.join(essay))
    
    # reading the annotations of the essays in SG2017 format 
    essays_ann = []
    for file in sorted(glob.glob( os.path.join(ann_dir,"*.ann") )):
        essay = open(file).readlines()
        essays_ann.append(essay)


    # ignoring the relations, keeping only the argument components labeling: MajorClaim, Claim, Premise
    # in addition to the segment's start and end character id (w.r.t the whole essay), and the actual argumentative segment
    essays_segments = []
    for essay in essays_ann:    
        segments = []
        
        for line in essay:
            # ignoring relations, i.e. line[0]=='R'
            if line[0] == 'T':
                _, label_s_e, text = line.rstrip().split('\t')
                label, start, end = label_s_e.split()

                # 4-tuple: label, start_char, end_char, and segment
                segments.append((label, int(start), int(end), text))
        
        # sorting segments start character id        
        segments.sort(key = lambda element : element[1])
        essays_segments.append(segments)



    return essays_txt_prg_list, essay_txt_str, essays_segments


def read_wm_essays(essays_dir):
     # read files
    data = []
    for i, file in enumerate(sorted(glob.glob( os.path.join(essays_dir,"*.tsv") ))):
        print(i, file)
        data.append(open(file).readlines())


    essays_sent_token_label, tokens, labels = [], [], []

    for essay in data:
        prev_sent_id = '0'
        essay_sents, sent_token, sent_label = [], [], []
        doc_tokens, doc_labels = [], []
        
        # assuming header in the first line
        for line in essay[1:]:
            sent_id, token_id, token, label = line.rstrip().split()
            
            if sent_id != prev_sent_id:
                essay_sents.append((sent_token, sent_label))
                sent_token, sent_label = [], []
                
            if len(token) < 25 and 'www' not in token:
                doc_tokens.append(token)
                # doc_labels.append('{}-claim'.format(label.split('-')[0]))
                doc_labels.append(label)
                sent_token.append(token)
                # sent_label.append('{}-claim'.format(label.split('-')[0]))
                sent_label.append(label)
            
            prev_sent_id = sent_id
        
        essay_sents.append((sent_token, sent_label))
        essays_sent_token_label.append(essay_sents)
        tokens.append(doc_tokens)
        labels.append(doc_labels)

    essay_str, essay_str_sent = [], []
    for essay in essays_sent_token_label:
        
        sentences = []
        for essay_sent_tokens, essay_sent_labels in essay:
            sent = ' '.join(essay_sent_tokens)
            sentences.append(sent)
        
        essay_str_sent.append(sentences)
        essay_str.append(' '.join(sentences))

    return essays_sent_token_label, tokens, labels, essay_str, essay_str_sent

