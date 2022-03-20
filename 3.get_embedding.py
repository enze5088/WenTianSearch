import csv
import sys

import torch
from tqdm import tqdm

sys.path.append("..")
from .simcse.models import BertForCL
from transformers import AutoTokenizer

device = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
batch_size = 100
use_pinyin = False


def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings


if __name__ == '__main__':
    model = BertForCL.from_pretrained("./result/unsup-simcse/")
    model.to(device)
    corpus = [line[1] for line in csv.reader(open("./data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/dev.query.txt"), delimiter='\t')]

    query_embedding_file = csv.writer(open('./query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 200001, writer_str])

    doc_embedding_file = csv.writer(open('./doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            doc_embedding_file.writerow([i + j + 1, writer_str])
