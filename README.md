# BEauRTy: BERT for Sequence Classification of Beauty Reviews

##  Overview
The BERT model was proposed in [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova](https://arxiv.org/abs/1810.04805). It’s a bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia.

The abstract from the paper is the following:

>We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
>
>BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

Tips:
   - BERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
   - BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.

> Source: [Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert)

## Requirements:

This project uses Python 3.6+

Create a virtual environment with:

```bash
python3 -m virtualenv venv
source venv/bin/activate
```

Install the requirements (inside the project folder):
```bash
(venv) pip3 install -r requirements.txt
```

## Getting Started:

### Train the model:
```bash
(venv) python run_beauty_reviews.py --do_train
```

Available commands:

Training arguments:
```bash
optional arguments:
  --seed                          Training seed.
  --per_device_train_batch_size   Batch size to be used at training.
  --per_device_eval_batch_size    Batch size to be used at training.
  --model_name_or_path            Model to use
```

See more arguments at `run_beauty_reviews.py` and [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) from HuggingFace.


**Note:**
After BERT several BERT-like models were released. You can test different size models like Mini-BERT and DistilBERT which are much smaller.
- Mini-BERT only contains 2 encoder layers with hidden sizes of 128 features. Use it with the flag: `--model_name_or_path google/bert_uncased_L-2_H-128_A-2`
- DistilBERT contains only 6 layers with hidden sizes of 768 features. Use it with the flag: `--model_name_or_path distilbert-base-uncased`

Training command example:
```bash
python run_beauty_reviews.py \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --model_name_or_path google/bert_uncased_L-2_H-128_A-2 \
    --do_train \
    --do_eval
```

This will create a folder containing the model fine-tuned on the beauty dataset at `experiments/`.

Testing the model:
```bash
python run_beauty_reviews.py \
     --model_name_or_path experiments/experiment_%Y-%m-%d_%H-%M-%S \
     --do_predict \
     --test_file data/test.txt
```

This command generates a `predictions.txt` file inside the experiment's folder with the examples from the test dataset.

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/"
```

### References
- [Minimalist BERT implemetation](https://github.com/ricardorei/lightning-text-classification) from [@ricardorei](https://github.com/ricardorei)
- [Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert)
