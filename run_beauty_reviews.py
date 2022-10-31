# coding=utf-8
""" Finetuning the library models for classification on the Beauty dataset (Bert, Roberta, XLNet)."""


import logging
import os, json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from datetime import datetime

import torch
import numpy as np

from transformers import (
    BertConfig,
    BertTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
    set_seed,
)

from transformers.integrations import TensorBoardCallback

from models.models import BertForSequenceClassification
from models.datasets import load_beauty_dataset


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class CustomTrainingArguments(TrainingArguments):
    num_train_epochs: float = field(default=20.0, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(
        default="experiments/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    metric_for_best_model: Optional[str] = field(
        default="acc", 
        metadata={"help": "The metric to use to compare two different models."}
    )
    report_to: Optional[List[str]] = field(
        default="all", 
        metadata={"help": "The list of integrations to report the results and logs to."}
    )
    evaluation_strategy: IntervalStrategy = field(
        default='no',
        metadata={"help": "The evaluation strategy to use."},
    )
    patience: Optional[int] = field(
        default=5,
        metadata={"help": "The number of patience validations to stop training after no improvement."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/bert_uncased_L-2_H-128_A-2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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

 
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_file: str = field(
        metadata={"help": "Should contain the data files for the task."},
        default="data/train.txt"
    )
    lowercase: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to lowercase the dataset."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    test_file: str = field(
        metadata={"help": "Should contain the data files for the task."},
        default=None,
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "experiment_" in model_args.model_name_or_path:
        training_args.output_dir = model_args.model_name_or_path
    else:
        training_args.output_dir = os.path.join(training_args.output_dir, "experiment_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if training_args.do_predict and not data_args.test_file:
        raise ValueError(
            f"--do_predict is set to True but no test file was provided. Use --test_file to provide a path for the test file."
        )

    # Set evaluation strategy if not set
    if training_args.do_eval: 
        training_args.load_best_model_at_end = True
        if training_args.evaluation_strategy == 'no':
            logger.warning("Flag do_eval was set but no --evaluation_strategy was provided. Using evaluation_strategy = epoch.")
            training_args.evaluation_strategy = 'epoch'
            training_args.save_strategy = 'epoch'
        else:
            training_args.save_strategy = training_args.evaluation_strategy
        
        
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    dataset = load_beauty_dataset(
        data_args.data_file,
        tokenizer,
        lowercase=data_args.lowercase,
        overwrite_cache=data_args.overwrite_cache,
        max_length=data_args.max_seq_length,

    )

    datasets = dataset.train_test_split(test_size=0.1)

    if training_args.do_train:
        train_dataset = datasets['train']
        label_list = train_dataset.features["label"].names
        
    if training_args.do_eval:
        eval_dataset = datasets['test']
        label_list = eval_dataset.features["label"].names

    if training_args.do_predict:
        test_dataset = load_beauty_dataset(
            data_args.test_file,
            tokenizer,
            lowercase=data_args.lowercase,
            overwrite_cache=data_args.overwrite_cache,
            max_length=data_args.max_seq_length,
        )
        label_list = test_dataset.features["label"].names

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(set(dataset['label'])),
        finetuning_task="BEAUTY",
        cache_dir=model_args.cache_dir,
    )
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.patience), TensorBoardCallback]
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(asdict(data_args), os.path.join(training_args.output_dir, "data_args.bin"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
    
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("Label ||| Prediction\t\tExample\n")
                for index, item in enumerate(predictions):
                    label = label_list[test_dataset[index]['label']]
                    item = label_list[item]
                    text = test_dataset[index]['text']
                    writer.write(f"{label} ||| {item}\t\t{text}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()