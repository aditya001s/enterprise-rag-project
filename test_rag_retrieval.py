#!/usr/bin/env python
"""
Quick test of RAG system without generation (only retrieval)
Useful for verifying GPU/CPU setup without downloading large models
"""
import argparse
import sys
import json
import os
from sentence_transformers import SentenceTransformer
from rag_retrieve import retrieve_with_faiss
from gpu_utils import get_device

def main():
    parser = argparse.ArgumentParser(description='Quick RAG retrieval test')
    parser.add_argument('--query', default='What are access control requirements?', help='Test query')
    parser.add_argument('--k', type=int, default=3, help='Number of results')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    device = get_device(use_gpu=args.gpu)
    device_str = str(device)
    
    base = os.path.dirname(__file__)
    index_path = os.path.join(base, 'faiss_index.index')
    meta_path = os.path.join(base, 'faiss_index_meta.json')
    chunks_path = os.path.join(base, 'extracted_chunks.csv')

    # Check files
    for path in [index_path, meta_path, chunks_path]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            print("Run 'python build_faiss_index.py' first")
            sys.exit(1)

    print(f"\nDevice: {device_str}")
    print(f"Query: {args.query}")
    print(f"Results: {args.k}\n")

    # Load embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device_str)

    # Retrieve
    print("Retrieving relevant chunks...")
    results = retrieve_with_faiss(args.query, index_path, meta_path, chunks_path, embed_model, args.k, device=device_str)

    if not results:
        print("No results found!")
        return

    # Display results
    print(f"\nTop {args.k} Results:\n")
    for r in results:
        print(f"{'='*80}")
        print(f"Rank: {r['rank']} | Score: {r['score']:.4f}")
        print(f"Source: {r['source']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"-" * 80)
        print(f"{r['snippet'][:500]}...")
        print()

if __name__ == '__main__':
    main()