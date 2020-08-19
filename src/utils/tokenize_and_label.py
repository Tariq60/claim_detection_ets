import spacy



def get_labels(	essay_spacy, 	# an essay processed by spacy, expected be list of 'spacy' tokens and have '.sents' attribute
				segments,		# labeled segments, 4-tuple list that has: Type_of_Arg_segment ('Claim', 'MajorClaim' or 'Premise'),
							 	# start_char, end_char, segment
				labels_as_numbers=True
			   ):
    '''	Returns either integer or string labels: O = 0, Arg-B = 1, Arg-I = 2
    	tokens in MajorClaim, Claim and Premise segments are tagged as either Arg-B and Arg-I, with no distinction'''
    
    doc_len = len(essay_spacy)
    
    labels = []
    tokens = []
    arg_seg_starts = [start for arg_type, start, end, text in segments]
    
    for token in essay_spacy:
        arg_I_token = False

        if token.idx in arg_seg_starts:
        	if labels_as_numbers:
        		labels.append(1.0)
        	else:
        		labels.append('Arg-B')

        	tokens.append(token.text)
        	assert token.text in segments[arg_seg_starts.index(token.idx)][-1]
        
        else:
            
            for _, start, end, _ in segments:
                if token.idx > start and token.idx+len(token) <= end:
                    if labels_as_numbers:
                    	labels.append(2.0)
                    else:
                    	labels.append('Arg-I')
                    tokens.append(token.text)
                    arg_I_token = True
            
            if not arg_I_token:
                if labels_as_numbers:
                	labels.append(0.0)
                else:
                	labels.append('O')
                tokens.append(token.text)

    assert len(labels) == doc_len
    return tokens, labels


def get_labels_claim_premise(essay_spacy, 	# an essay processed by spacy, expected be list of 'spacy' tokens and have '.sents' attribute
							 segments,		# labeled segments, 4-tuple list that has: Type_of_Arg_segment ('Claim', 'MajorClaim' or 'Premise'),
							 				# start_char, end_char, segment
							 mode='claim', 	# string in one of the following: claim, premsie or all
							 labels_as_numbers=True # returns numbers intead of str, will be added later
							):
	'''	Returns labeled tokens of claims, premises, or both, according to the selected mode
    	labels are: B-claim, I-claim, B-premise, I-premise, O'''
    
    # modes are: claim, premise, all
	assert mode in ['claim', 'premise', 'all']
    
    # including argument types that correspond to each mode
	mode_labels= {'claim':['Claim','MajorClaim'], 'premise':'Premise', 'all': ['Claim', 'MajorClaim', 'Premise']}

	# tagging based on argument type
	arg_type_to_tag = {'MajorClaim': 'claim', 'Claim': 'claim', 'Premise': 'premise'}

	doc_len = len(essay_spacy)
	labels, tokens = [], []

	# getting list of start character indecies of argumentative segments to be used for tagging B-xxxx segments 
	arg_seg_starts = [start for arg_type, start, end, text in segments if arg_type in mode_labels[mode]]
	arg_seg_arg_type = [arg_type for arg_type, start, end, text in segments if arg_type in mode_labels[mode]]
	arg_seg_texts = [text for arg_type, start, end, text in segments if arg_type in mode_labels[mode]]

	# Looping through the tokenized essay
	for token in essay_spacy:

		# initializing the 'found Arg-I/I-claim/I-premise' flag to false for each token
		arg_I_token = False

		# B-type tagging: checking if the starting index of the token is in the list of all arg segment starts in the essay
		if token.idx in arg_seg_starts:
			if labels_as_numbers:
				labels.append(1.0)
			else:
				labels.append('B-' + arg_type_to_tag[arg_seg_arg_type[arg_seg_starts.index(token.idx)]])
			tokens.append(token.text)

			# verifying that the token is in the argumentative text segment
			assert token.text in arg_seg_texts[arg_seg_starts.index(token.idx)]

		else:
			# checking if token is inside any of the segments of the chosen arg_type mode, i.e. claim, premise, or both
			for arg_type, start, end, _ in segments:
				if arg_type in mode_labels[mode]:
					if token.idx > start and token.idx+len(token) <= end:
						if labels_as_numbers:
							labels.append(2.0)
						else:
							labels.append('I-'+arg_type_to_tag[arg_type])
						tokens.append(token.text)
						arg_I_token = True


						# A break statement could be inserted here to increase speed

	# Since segments are sorted, another way to speed this up, in case of bigger datasets or longer essays is by:
	#	defining an iterator over segments and only looping from: segments[prev_token_iter_break_point], to: end or until token is found
            

            # if token is not in any segment then it is an O-type token
			if not arg_I_token:
				if mode == 'claim':
					# token is labeled as outside-claim in claim mode
					if labels_as_numbers:
						labels.append(0.0)
					else:
						labels.append('O-claim')
				elif mode == 'premise':
					# token is labeled as outside-premise in premie mode
					if labels_as_numbers:
						labels.append(0.0)
					else:
						labels.append('O-premise')
				else:
					# token is labeled as other/outside-argument in 'all' mode
					if labels_as_numbers:
						labels.append(0.0)
					else:
						labels.append('O')
				tokens.append(token.text)
	
	# asserting that all tokens are labeled
	assert len(tokens) == len(labels) == doc_len

	return tokens, labels
