import os
import gzip
import csv
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
import math
import logging
from pathlib import Path
from typing import List


#Configuration
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 32
NUM_EPOCHS = 5
DATA_PATH = Path('datasets/stsbenchmark.tsv.gz')
CROSS_ENCODER_PATH = Path('models/cross_encoder_model')
BI_ENCODER_PATH = ('models/bi_encoder_model_after_cross_encoder')
LOG_PATH = Path("logs/semi_supervised_bi_encoder.log")
TOP_K = 3 #similar sentences to retrieve

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
    train, validation = [],[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        for row in reader:
            # Ensuring symmetric pairs, i.e. CrossEncoder(A,B) = CrossEncoder(B,A)
            # This ensure that each pair in the training dataset is symmetric, as cross encoder should generate the same similarity score for them
            example1 = InputExample(texts=[row['sentence1'],row['sentence2']],label = float(row['score'])/5.0)
            example2 = InputExample(texts=[row['sentence2'],row['sentence1']],label=float(row['score'])/5.0)
            if row['split'] == 'dev':
                validation.append(example1)
            if row['split'] == 'train':
                train.append(example1)
                train.append(example2)
    return train,validation

def train_cross_encoder_model(train_data: List[InputExample],validation_data: List[InputExample]):
    #Initialize the cross-encoder model
    cross_encoder = CrossEncoder(MODEL_NAME,num_labels=1)
    #create a DataLoader for the training data
    train_dataloader = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
    #set up an evaluation to monitor progress on the validation set
    evaluator = CrossEncoderCorrelationEvaluator.from_input_examples(validation_data, name = 'sts-dev')

    #train the cross-encoder model
    cross_encoder.fit(train_dataloader=train_dataloader,
                      evaluator=evaluator,
                      epochs=NUM_EPOCHS,
                      evaluation_steps=1000,
                      output_path=str(CROSS_ENCODER_PATH))
    return cross_encoder

def create_silver_pairs_for_labeling(train_data: List[InputExample]):
    augmented = [] #to store new sentence pairs
    sentences = set() #to store unique sentences from training dataset

    for sample in train_data:
        sentences.update(sample.texts)
    #converting to list and create a dictionary to map each sentence to a unique index
    sentences = list(sentences)
    sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)}
    # avoid pairs that already exist in train_examples
    duplicates = set((sent2idx[data.texts[0]],sent2idx[data.texts[1]]) for data in train_data)
    # duplicates contains the indexes of the sentence pairs that already exist in the training data
    #using pre-trained Bi-encoder model to find similar sentences
    # model bert-base-nli-stsb-mean-tokens model is used since its trained on STS benchmark

    #using pre-trained model for semantic search
    semantic_search_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    #encoding all the unqiue sentences
    embeddings = semantic_search_model.encode(sentences,batch_size=BATCH_SIZE,convert_to_tensor=True)
    
    #computing cosine similarity with each sentence in the dataset and retrieving top-k most similar sentences
    for idx in tqdm(range(len(sentences)),unit="docs"):
        sentence_embedding = embeddings[idx]
        cos_scores = util.cos_sim(sentence_embedding,embeddings)[0]
        print(cos_scores)
        cos_scores = cos_scores.cpu()

        #retrieve the top-k most similar sentences (excluding the sentence itself)
        top_results = torch.topk(cos_scores,k=TOP_K+1) #adding 1 to remove similarity from its own value
        # print(top_results.shape())
        # print(top_results)

        for score,iid in zip(top_results[0],top_results[1]):
            # top_results[0] provide the values
            # top_results[1] provide the indices
            iid = iid.item()
            if iid != idx and (iid,idx) not in duplicates:
                augmented.append((sentences[idx],sentences[iid]))
                duplicates.add((idx,iid))
    return augmented
def labeling_data(augmented_data):
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)
    silver_scores = cross_encoder.predict(augmented_data)
    #prepare silver samples in the required format
    aug_samples = [InputExample(texts = [pair[0],pair[1]],label=score) for pair,score in zip(augmented_data,silver_scores)]
    return aug_samples
def build_bi_encoder_model(model_name: str) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])
def train_bi_encoder_model(train_data: List[InputExample], validation_data: List[InputExample], bi_encoder_model: SentenceTransformer):
    logging.info("Starting model training...")
    train_dataset = SentencesDataset(train_data, bi_encoder_model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model=bi_encoder_model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_data, name="sts-dev")

    bi_encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=1000,
        output_path=str(BI_ENCODER_PATH)
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_dataset(DATA_PATH)
    train_examples, val_examples = load_dataset(DATA_PATH)
    trained_cross_encoder = train_cross_encoder_model(train_examples,val_examples)
    augmented = create_silver_pairs_for_labeling(train_examples)
    silver_data_label = labeling_data(augmented)
    bi_encoder_model = build_bi_encoder_model(MODEL_NAME)
    train_bi_encoder_model(train_examples + silver_data_label, val_examples, bi_encoder_model)