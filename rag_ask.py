#!/usr/bin/env python
import argparse
import sys
import json
import os
from sentence_transformers import SentenceTransformer
from rag_retrieve import retrieve_with_faiss
from rag_generate import generate_response
from gpu_utils import get_device, clear_gpu_memory

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=3, help='Number of passages to retrieve')
    p.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
    p.add_argument('--gen-model', default='gpt2', help='Generative model')
    p.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = p.parse_args()

    device = get_device(use_gpu=args.gpu)
    device_str = str(device)

    base = os.path.dirname(__file__)
    index_path = os.path.join(base, 'faiss_index.index')
    meta_path = os.path.join(base, 'faiss_index_meta.json')
    chunks_path = os.path.join(base, 'extracted_chunks.csv')

    # Check files
    for path in [index_path, meta_path, chunks_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required file not found: {path}. Run build_faiss_index.py first.')

    embed_model = SentenceTransformer(args.embed_model, device=device_str)

    print('Interactive RAG. Enter a question (empty to quit).')
    while True:
        try:
            q = input('\nQuestion: ').strip()
        except EOFError:
            break
        if not q:
            break
        
        # Retrieve
        context_chunks = retrieve_with_faiss(q, index_path, meta_path, chunks_path, embed_model, args.k, device=device_str)
        print('\nRetrieved chunks:')
        for r in context_chunks:
            print(f"[{r['rank']}] (score={r['score']:.4f}) {r['source']} chunk={r['chunk_id']}")
            print(f"{r['snippet']}\n")

        # Generate
        print('Generating response...')
        try:
            response = generate_response(q, context_chunks, args.gen_model, device=device_str)
            print('Generated Response:')
            print(response)
        except Exception as e:
            print(f'Generation failed: {e}')

if __name__ == '__main__':
    main()