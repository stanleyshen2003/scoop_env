import numpy as np
from sentence_transformers import SentenceTransformer

def embed(x, model_name="sentence-transformers/paraphrase-distilroberta-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(x)

def retrieve(sentence, train_file, splitter='='*10, top_k=5):
    train_data = np.asanyarray(''.join(open(train_file, 'r').readlines()).split(splitter))
    sen_emb = embed([sentence])
    train_emb = embed(train_data)
    sims = np.dot(sen_emb, train_emb.T).squeeze()
    topk_idx = np.argsort(-sims)[:top_k if top_k < len(train_data) else len(train_data)]
    return train_data[topk_idx]

if __name__ == "__main__":
    sentence = 'I am a student' 
    train_file = 'train.txt'
    print(retrieve(sentence, train_file, splitter='\n', top_k=3))