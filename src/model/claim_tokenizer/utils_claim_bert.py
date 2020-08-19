# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


# class Split(Enum):
#     train = "train"
#     dev = "dev"
#     test = "test"


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    
    class ClaimDataset(Dataset):

        def __init__(self, data_dir, tokenizer, max_seq_length, mode):
            self.examples = read_examples_from_file(data_dir, mode)
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            
            pad_tok = tokenizer.vocab["[PAD]"]
            sep_tok = tokenizer.vocab["[SEP]"]
            cls_tok = tokenizer.vocab["[CLS]"]
            
            label_list = self.get_labels()  # [0, 1] for binary classification
            label_map = {label: i+1 for i, label in enumerate(label_list)}
            
            examples_for_processing = [(example, label_map, self.max_seq_length,\
                                          self.tokenizer, 'classification') \
                                         for example in self.examples]
            self.features = []
            for example in examples_for_processing:
                feature = self.convert_examples_to_features(example)
                self.features.append(feature)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return self.features[i]




def read_examples_from_file(data_dir, mode) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples




def convert_examples_to_features(self, claim_example):
    
    pad_token_label_id = 0
    example, label_map, max_seq_length, tokenizer, output_mode = claim_example
    tokens = []
    label_ids = []
    
    for word, label in zip(example.words, example.labels):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        #label_ids.extend([label_map.get(label)] + [pad_token_label_id] * (len(word_tokens) - 1))
        label_ids.extend([label_map.get(label)] + [label_map.get(label)] * (len(word_tokens) - 1))
    
    assert len(tokens) == len(label_ids)
    
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
        label_ids = label_ids[:(max_seq_length - 2)]
    
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
    segment_ids = [0] * len(tokens)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
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
    
    
    return InputFeatures(input_ids=input_ids,
                         attention_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids)



def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["B-claim", "I-claim", "O-claim"]

    
    
    
    
    
    
    