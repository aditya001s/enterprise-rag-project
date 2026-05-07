import argparse
import os
import json
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from gpu_utils import get_device, clear_gpu_memory

def retrieve_with_faiss(query, index_path, meta_path, chunks_path, model, k=3, device='cpu'):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    index = faiss.read_index(index_path)
    
    # Encode query with device placement
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # Search
    scores, indices = index.search(query_emb, k)
    
    # Load metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    
    # Load chunks
    chunks_df = pd.read_csv(chunks_path)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:  # No more results
            break
        meta = metas[idx]
        chunk_text = chunks_df.iloc[idx]['text']
        snippet = chunk_text.replace('\n', ' ')[:1000]
        results.append({
            'rank': rank,
            'score': float(score),
            'source': meta['source'],
            'chunk_id': meta['chunk_id'],
            'snippet': snippet,
            'full_text': chunk_text
        })
    
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True, help='Query string')
    p.add_argument('--index', default='faiss_index.index', help='FAISS index path')
    p.add_argument('--meta', default='faiss_index_meta.json', help='Metadata JSON path')
    p.add_argument('--chunks', default='extracted_chunks.csv', help='Chunks CSV path')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
    p.add_argument('--k', type=int, default=3, help='Number of results')
    p.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = p.parse_args()
    
    device = 'cuda' if args.gpu else 'cpu'
    model = SentenceTransformer(args.model, device=device)
    results = retrieve_with_faiss(args.query, args.index, args.meta, args.chunks, model, args.k, device=device)
    
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()