from transformers import AutoConfig, EvalPrediction, AutoTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from claim_data_handler import ClaimDataset,ClaimExample,_load_data,_load_each_file
from bert_training import Trainer
from transformers import DistilBertConfig
from torch.utils.data import Dataset
from typing import Dict, Optional
from utils import f1, store_preds
from bert_claim_identifier_model import AutoModelForClaimTokenClassification
import numpy as np

import torch
import os
import logging
import glob
from dataclasses import dataclass, field
from typing import Dict, Optional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files \
        (or other data files) for the task.", "default":'/Users/dghosh/work/claim_detection_workspace/claim_detection_wm/data/input/sg2017/tokens/train/'}
    )
    train_file: str
    eval_file: str
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    # overwrite_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier\
         from huggingface.co/models","default":'bert_bl_cased'}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def _use_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


def main():

    #label_types = ['B','I','O']
    #rtag2idx = {t: i for i, t in enumerate(label_types)}


    #_use_cuda()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=3,
    )

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )

    #model = AutoModelForClaimTokenClassification.from_pretrained(
    #    model_args.model_name_or_path,
    #    config=config,
    #)
    lm_path = '/home/nlp-text/dynamic/dghosh/bert_workspace/data/output/imho_transformer_epoch10/'
    lm_model = 'pytorch_model.bin'
    #lm_path = '/home/nlp-text/dynamic/dghosh/bert_workspace/data/output/essays_score3_lm3_pytorch_epoch7/'
    #lm_model = 'pytorch_model_5.bin'

    lm_state_dict = torch.load(os.path.join(lm_path, lm_model))
    model = AutoModelForClaimTokenClassification.from_pretrained(
        model_args.model_name_or_path, config=config,
        state_dict=lm_state_dict,
    )


    # Fetch Datasets
    train_set = ClaimDataset(_load_data(os.path.join(data_args.data_dir,'wm2020/claim_tokens/train/')  ,data_mode='individual'), \
                             tokenizer, training_args,'training') if training_args.do_train else None
    valid_set = ClaimDataset(_load_data(os.path.join(data_args.data_dir,'sg2017/claim_tokens/valid/'), data_mode='individual'), \
                             tokenizer, training_args,'valid') if training_args.do_train else None
    eval_set = ClaimDataset(_load_data(os.path.join(data_args.data_dir,'wm2020/claim_tokens/eval/'), data_mode='individual'), \
                            tokenizer, training_args,'eval') if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        #preds = np.argmax(p.predictions, axis=1)
        #preds = p.predictions
        return f1(p.predictions, p.label_ids)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_datasets = [eval_set]
        for eval_dataset in eval_datasets:
            result_set = trainer.evaluate(eval_dataset=eval_dataset)
            result = result_set.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_claim_tokens.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results Alternate *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    print(value)

            results.update(result)

            preds, label_ids = result_set.predictions, result_set.label_ids
            #preds, labels = store_preds(EvalPrediction(predictions=preds, label_ids=label_ids))

            '''
            data = _load_data(data_args, evaluate=True)
            context, reply = [], []
            for example in data:
                ctx, rpl = example.split('\t')[0:2]
                context.append(ctx)
                reply.append(rpl)

            output_score_file_t1 = os.path.join(
                training_args.output_dir, f"eval_preds_t1_alternate.txt"
            )
            '''

            '''
            with open(output_score_file_t1, "w") as writer:
                for i in range(len(labels)):
                    writer.write("%s\t%s\t%s\t%s\n" % (context[i], reply[i], labels[i], preds[i]))
            '''

    return results


if __name__ == "__main__":
    main()
