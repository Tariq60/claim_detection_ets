from dataclasses import dataclass, field
from typing import Dict, Optional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import os
import logging
import glob

from utils import calculate_class_weights

def _load_each_file(input_file,mode='train'):
    file_name = os.path.basename(input_file)
    print(file_name)
    examples = []
    guid_index = 1
    previous_sentence_id = 0
    with open(input_file) as f:
        words = []
        labels = []
        header = f.readline()
        for line in f:
            #print(line.strip())
            sentence_id,word_id,word,label = line.strip().split('\t')
            if int(sentence_id) !=previous_sentence_id:
                '''new example'''
                examples.append(ClaimExample(guid="{}-{}-{}".format(mode, file_name, guid_index),
                                             mode=mode,\
                                             file_name=file_name,\
                                                 words=words,\
                                                 labels=labels))
                guid_index+=1
                words = []
                labels = []
                previous_sentence_id = int(sentence_id)
                words.append(word)
                labels.append(label)
            else:
                '''continuing'''
                words.append(word)
                labels.append(label)

        if words:
            examples.append(ClaimExample(guid="{}-{}-{}".format(mode, file_name, guid_index),\
                                         mode=mode,\
                                         file_name=file_name,\
                                         words=words,\
                                         labels=labels))
    return examples

def _paired_examples(examples):
    example_pairs = []
    index = 0
    while True:

        if index >= len(examples):
            break

        if index < len(examples):
            example1 = examples[index]
            index+=1
        if index < len(examples):
            example2 = examples[index]
        if example1 is not None and example2 is not None:
            example_pairs.append(ClaimPairExample(guid="{}-{}".format(example1.mode, example1.file_name\
                                            ),example1=example1,\
                                           example2=example2))

    return example_pairs

def _load_data(data_dir,data_mode='individual'):

    all_examples = []
    input_files = glob.glob(data_dir+'/*.tsv')
    print('{}{}{}'.format('input_dir:','\t',data_dir))

    for input_file in input_files:
        if data_mode == 'pair':
            all_examples.extend(_paired_examples(_load_each_file(input_file)))
        if data_mode == 'individual':
            all_examples.extend(_load_each_file(input_file))

    print(len(all_examples))
    return all_examples

class ClaimPairExample(object):

    """A pair training/test example for token classification."""

    def __init__(self, guid, example1, example2):
        """Constructs a ClaimExample.
        Args:
            guid: Unique id for the example.
            example1: first example
            example2: second example
        """
        self.guid = guid
        self.example1 = example1
        self.example2 = example2


class ClaimExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, mode, file_name,words, labels):
        """Constructs a ClaimExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.mode = mode
        self.file_name = file_name
        self.words = words
        self.labels = labels


class ClaimFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = segment_ids
        self.label_ids = label_ids


class ClaimDataset(Dataset):
    def __init__(self, examples, tokenizer, args, mode='training'):
        self.examples = examples
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = 256

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        cls_tok = tokenizer.vocab["[CLS]"]

        label_list = self.get_labels()  # [0, 1] for binary classification

        label_map = {label: i for i, label in enumerate(label_list)}
        examples_for_processing = [(example, label_map, self.max_seq_length,\
                                      self.tokenizer, 'classification') \
                                     for example in self.examples]

        self.features = []
        for example in examples_for_processing :
            feature = self.convert_examples_to_features(example)
            self.features.append(feature)

        #class weights
        if mode == 'training':
            self.class_weight = calculate_class_weights(self.features)

    def convert_examples_to_features(self,claim_example):
        pad_token_label_id = -100
        mask_padding_with_zero = True
        example, label_map, max_seq_length, tokenizer, output_mode = claim_example

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                #label_ids.extend([label_map.get(label)] + [label_map.get(label)] * (len(word_tokens) - 1))

        assert len(tokens) == len(label_ids)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            label_ids = label_ids[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #input_mask = [1] * len(input_ids)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        padding = [0] * (padding_length)
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        '''
        if output_mode == "classification":
            label_ids = [label_map.get(str(label_id)) for label_id in label_ids]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        '''
        return ClaimFeature(input_ids=input_ids,
                             attention_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids)

    def get_labels(self):
        return ['O-claim','B-claim','I-claim']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
