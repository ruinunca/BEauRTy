import os
import re
import logging

from pathlib import Path

from filelock import FileLock
from datasets import Dataset

import torch


logger = logging.getLogger(__name__)

##############
# Beauty Dataset
##############


def load_beauty_dataset(data_file, tokenizer, lowercase=True, overwrite_cache=False, max_length=512):

    data_dir = os.path.dirname(data_file)

    file_name = Path(data_file).stem

    cached_features_file = os.path.join(
        data_dir,
        "cached_BEAUTY_{}_{}".format(file_name, tokenizer.__class__.__name__),
    )

    # Make sure only the first process in distributed training processes the dataset,
    # and the others will use the cache.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            dataset = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")

            with open(data_file) as f:
                data = f.readlines()

            raw_dataset = []
            for line in data:
                if len(line) == 0:
                    continue
                row = line.split('\t')
                raw_dataset.append({'label': row[0], 'text': row[1]})

            dataset = Dataset.from_list(raw_dataset)
            dataset = dataset.class_encode_column('label')
            print("Labels: ", dataset.features["label"].names)
            print(dataset)

            def preprocess_beauty(example):
                example['text'] = example['text'].lower()

            if lowercase:
                dataset = dataset.map(preprocess_beauty, num_proc=4)

            def tokenization(example):
                return tokenizer(example['text'], truncation=True, max_length=max_length)

            dataset = dataset.map(tokenization, batched=True, num_proc=4)

            logger.info("Training examples: %s", len(dataset))
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(dataset, cached_features_file)
    
    return dataset
