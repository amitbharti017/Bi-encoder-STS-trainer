import os
from pathlib import Path
from sentence_transformers import SentencesDataset, SentenceTransformer, util,models,losses,InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
import logging
from typing import List
import gzip
import csv
import torch
from tqdm import tqdm
import json
from datasets import load_dataset

#loading the config.json file
with open('config.json','r') as f:
    config = json.load(f)

BATCH_SIZE = config["BATCH_SIZE"]
DATA_PATH_STS = Path(config["DATA_PATH_STS"])
TRAIN_DATA_PATH_QUORA = Path(config["TRAIN_DATA_PATH_QUORA"])
CROSS_ENCODER_MODEL = config["CROSS_ENCODER_MODEL"]
LOG_PATH = Path(config["LOG_PATH"])
NUM_EPOCHS = config["NUM_EPOCHS"]
CROSS_ENCODER_TRAINED_MODEL = Path(config["CROSS_ENCODER_TRAINED_MODEL"])
BI_ENCODER_MODEL = config["BI_ENCODER_MODEL"]
TOP_K = config["TOP_K"]
BI_ENCODER_TRAINED_MODEL = config["BI_ENCODER_TRAINED_MODEL"]

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

def download_sts_dataset(path: Path):
    if not path.exists():
        logging.info(f"{path}  not found. Downloading...")
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', str(path))
    else:
        logging.info(f"{path} already exists.")

def download_quora_dataset(train_path: Path):
    if not train_path.exists():
        logging.info(f"{train_path}  not found. Downloading...")

        train_dataset = load_dataset("BeIR/quora", "corpus")
        corpus = train_dataset['corpus']
        texts = [item['text'] for item in corpus]
        # Optional: remove None or empty texts
        texts = [text for text in texts if text]
        #saving it locally
        train_path.parent.mkdir(parents=True, exist_ok=True)

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
        logging.info(f"Dataset downloaded and saved to {train_path}.")
    else:
        logging.info(f"Data in the {train_path} already exist")
def load_sts_dataset(path: Path):
    train,validation = [],[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        for row in reader:
            example1 = InputExample(texts=[row['sentence1'],row['sentence2']],label=float(row['score'])/5.0)
            example2 = InputExample(texts=[row['sentence2'],row['sentence1']],label=float(row['score'])/5.0)
            if row['split'] == 'train':
                train.append(example1)
                train.append(example2)
            if row['split'] == 'dev':
                validation.append(example1)
    return train,validation   

def train_cross_encoder_model(train_data: List[InputExample],validation_data:List[InputExample]):
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL,num_labels=1)
    train_data_loader_cross_encoder = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
    evaluator_cross_encoder = CrossEncoderCorrelationEvaluator.from_input_examples(validation_data,name='sts-dev')

    #training the cross-encoder model
    cross_encoder.fit(train_dataloader=train_data_loader_cross_encoder,
                      evaluator=evaluator_cross_encoder,
                      epochs=NUM_EPOCHS,
                      evaluation_steps=1000,
                      output_path=str(CROSS_ENCODER_TRAINED_MODEL))
    return cross_encoder

def load_quora_dataset(path: Path):
    with open(path,'r',encoding='utf-8') as f:
        quora_data = json.load(f)
    train_quora_data = list(set(quora_data))
    # train_quora_data = train_quora_data[:10]
    semantic_search_model = SentenceTransformer(BI_ENCODER_MODEL)
    embeddings = semantic_search_model.encode(train_quora_data,batch_size=BATCH_SIZE,convert_to_tensor=True)
    quora_two_sentences = set()
    for idx in tqdm(range(len(train_quora_data)),unit="docs"):
        sentence_embedding = embeddings[idx]
        cos_scores = util.cos_sim(sentence_embedding,embeddings)[0]
        cos_scores = cos_scores.cpu()
        top_results = torch.topk(cos_scores,k=TOP_K+1)
        for score,iid in zip(top_results[0],top_results[1]):
            iid = iid.item()
            if iid != idx:
                pair = tuple(sorted([train_quora_data[idx],train_quora_data[iid]]))
                quora_two_sentences.add(pair)
    quora_two_sentences = list(quora_two_sentences)
    length = len(quora_two_sentences)
    split_idx = int(0.8*length)
    train_quora = quora_two_sentences[:split_idx]
    validation_quora = quora_two_sentences[split_idx:]
    return train_quora, validation_quora

def weak_quora_labels(data):
    cross_encoder = CrossEncoder(CROSS_ENCODER_TRAINED_MODEL)
    silver_scores = cross_encoder.predict(data)
    train_data_bi_encoder = [InputExample(texts=[pair[0],pair[1]],label=score) for pair,score in zip(data,silver_scores)]
    return train_data_bi_encoder

def build_bi_encoder_model(bi_encoder_model: str) -> SentenceTransformer:
    word_embedding_model = models.Transformer(bi_encoder_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    return SentenceTransformer(modules=[word_embedding_model,pooling_model])

def train_bi_encoder_model(train_data: List[InputExample],validation_data: List[InputExample],bi_encoder_model:SentenceTransformer):
    logging.info(f"Starting bi-encoder training...")
    train_dataset = SentencesDataset(train_data,bi_encoder_model)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model = bi_encoder_model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_data,name="quora-dev")

    bi_encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=1000,
        output_path=str(BI_ENCODER_TRAINED_MODEL)
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_sts_dataset(DATA_PATH_STS)
    download_quora_dataset(TRAIN_DATA_PATH_QUORA)
    train_dataset_sts,validation_dataset_sts = load_sts_dataset(DATA_PATH_STS)
    train_cross_encoder_model(train_dataset_sts,validation_dataset_sts)
    train_dataset_quora = load_quora_dataset(TRAIN_DATA_PATH_QUORA)
    train_dataset_bi_encoder,validation_dataset_bi_encoder = weak_quora_labels(train_dataset_quora)
    bi_encoder_model = build_bi_encoder_model(BI_ENCODER_MODEL)
    train_bi_encoder_model(train_dataset_bi_encoder,validation_dataset_bi_encoder,bi_encoder_model)


