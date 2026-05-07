import argparse
import os
import numpy as np
import faiss
import json

def build_faiss_index(embeddings_path, index_path, meta_path):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    dimension = embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, index_path)
    
    # Load and save metadata
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metas = json.load(f)
        # Save metas alongside index
        meta_out = index_path.replace('.index', '_meta.json')
        with open(meta_out, 'w', encoding='utf-8') as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)
    
    print(f"Built FAISS index with {len(embeddings)} vectors, dimension {dimension}")
    print(f"Saved index to {index_path}")
    if os.path.exists(meta_path):
        print(f"Saved metadata to {meta_out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', default='embeddings.npy', help='Path to embeddings.npy')
    p.add_argument('--index', default='faiss_index.index', help='Output FAISS index path')
    p.add_argument('--meta', default='metas.json', help='Path to metas.json')
    args = p.parse_args()
    build_faiss_index(args.embeddings, args.index, args.meta)