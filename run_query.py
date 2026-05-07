#!/usr/bin/env python
import argparse
import os
import json
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--k', type=int, default=3)
    args = p.parse_args()

    base = os.path.dirname(__file__)
    emb_path = os.path.join(base, 'embeddings.npy')
    metas_path = os.path.join(base, 'metas.json')
    chunks_path = os.path.join(base, 'extracted_chunks.csv')

    if not os.path.exists(emb_path) or not os.path.exists(metas_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError('Required data files not found in training/. Run extract/embed first.')

    emb = np.load(emb_path)
    metas = pd.read_json(metas_path)
    chunks = pd.read_csv(chunks_path)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    emb_norm = emb / (norm(emb, axis=1, keepdims=True) + 1e-12)
    q_emb = model.encode([args.query], convert_to_numpy=True)
    qn = q_emb / (norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_norm @ qn[0]
    idxs = sims.argsort()[-args.k:][::-1]

    out = []
    for rank, idx in enumerate(idxs, 1):
        meta = metas.iloc[idx] if not metas.empty else {}
        text = chunks.iloc[idx]['text'] if 'text' in chunks.columns else ''
        snippet = text.replace('\n',' ')[:1000]
        item = {'rank': rank, 'score': float(sims[idx]), 'source': meta.get('source') if hasattr(meta, 'get') else None, 'chunk_id': int(meta.get('chunk_id')) if hasattr(meta, 'get') and meta.get('chunk_id') is not None else None, 'snippet': snippet}
        out.append(item)

    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
