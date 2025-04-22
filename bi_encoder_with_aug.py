import os
import gzip
import csv
import logging
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import nlpaug.augmenter.word as naw
from sentence_transformers import (
    util,
    InputExample,
    models,
    SentenceTransformer,
    SentencesDataset,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


# -------------------------
# Configuration
# -------------------------
DATA_PATH = Path("dataset/stsbenchmark.tsv.gz")
MODEL_NAME = "bert-base-uncased"
SAVE_PATH = Path("models/bi_encoder_with_aug")
LOG_PATH = Path("logs/bi_encoder_with_aug.log")
BATCH_SIZE = 32
NUM_EPOCHS = 5
AUGMENTATION_ACTION = "insert"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)


def download_dataset(path: Path):
    if not path.exists():
        logging.info(f"{path} not found. Downloading...")
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', str(path))
    else:
        logging.info(f"{path} already exists.")


def load_dataset(path: Path):
    train, validation = [], []
    with gzip.open(path, 'rt', encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            example = InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['score']) / 5.0)
            if row['split'] == 'train':
                train.append(example)
            elif row['split'] == 'dev':
                validation.append(example)
    return train, validation


def augment_data(examples: List[InputExample], model_name: str, device: str, action: str) -> List[InputExample]:
    logging.info("Starting data augmentation...")
    aug = naw.context_word_embs.ContextualWordEmbsAug(model_path=model_name, action=action, device=device)
    augmented = []
    for example in tqdm(examples, desc="Augmenting", unit="docs"):
        augmented_texts = aug.augment(example.texts)
        augmented.append(InputExample(texts=augmented_texts, label=example.label))
    return augmented


def build_model(model_name: str) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def train_model(train_data: List[InputExample], validation_data: List[InputExample], model: SentenceTransformer):
    logging.info("Starting model training...")
    train_dataset = SentencesDataset(train_data, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_data, name="sts-dev")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=1000,
        output_path=str(SAVE_PATH)
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    download_dataset(DATA_PATH)
    train_examples, val_examples = load_dataset(DATA_PATH)
    augmented_examples = augment_data(train_examples, MODEL_NAME, device, AUGMENTATION_ACTION)

    model = build_model(MODEL_NAME)
    train_model(train_examples + augmented_examples, val_examples, model)
