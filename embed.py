import argparse
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def compute_embeddings(csv_path, model_name, out_embeddings, out_meta, batch_size=64):
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
    np.save(out_embeddings, embeddings)
    metas = df[['source','chunk_id']].to_dict(orient='records')
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings ({embeddings.shape}) to {out_embeddings} and metas to {out_meta}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV of chunks (text,source,chunk_id)')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--out-emb', default='embeddings.npy')
    p.add_argument('--out-meta', default='metas.json')
    args = p.parse_args()
    compute_embeddings(args.csv, args.model, args.out_emb, args.out_meta)
