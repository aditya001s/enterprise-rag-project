#!/usr/bin/env python
import argparse
import os
import json
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from rag_retrieve import retrieve_with_faiss
from rag_generate import generate_response
from gpu_utils import get_device, clear_gpu_memory

def run_rag_pipeline(query, k=3, embed_model_name='sentence-transformers/all-MiniLM-L6-v2', gen_model_name='gpt2', use_gpu=True):
    base_dir = os.path.dirname(__file__)
    
    device = get_device(use_gpu=use_gpu)
    device_str = str(device)
    
    # Paths
    index_path = os.path.join(base_dir, 'faiss_index.index')
    meta_path = os.path.join(base_dir, 'faiss_index_meta.json')
    chunks_path = os.path.join(base_dir, 'extracted_chunks.csv')
    
    # Check if index exists, if not build it
    if not os.path.exists(index_path):
        print("FAISS index not found. Building index...")
        emb_path = os.path.join(base_dir, 'embeddings.npy')
        meta_orig_path = os.path.join(base_dir, 'metas.json')
        if not os.path.exists(emb_path) or not os.path.exists(meta_orig_path):
            raise FileNotFoundError("Embeddings or metadata not found. Run embed.py first.")
        cmd = [sys.executable, os.path.join(base_dir, 'build_faiss_index.py'), '--embeddings', emb_path, '--index', index_path, '--meta', meta_orig_path]
        subprocess.run(cmd, check=True)
        # Rename meta
        os.rename(os.path.join(base_dir, 'faiss_index_meta.json'), meta_path)
    
    # Load embed model
    embed_model = SentenceTransformer(embed_model_name, device=device_str)
    
    # Retrieve
    context_chunks = retrieve_with_faiss(query, index_path, meta_path, chunks_path, embed_model, k, device=device_str)
    
    if not context_chunks:
        return {'query': query, 'error': 'No relevant context found.'}
    
    # Generate
    response = generate_response(query, context_chunks, gen_model_name, device=device_str)
    
    clear_gpu_memory()
    
    return {
        'query': query,
        'retrieved_chunks': context_chunks,
        'generated_response': response
    }

def main():
    p = argparse.ArgumentParser(description='RAG Pipeline: Retrieve and Generate')
    p.add_argument('--query', required=True, help='User query')
    p.add_argument('--k', type=int, default=3, help='Number of chunks to retrieve')
    p.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
    p.add_argument('--gen-model', default='gpt2', help='Generative model')
    p.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = p.parse_args()
    
    result = run_rag_pipeline(args.query, args.k, args.embed_model, args.gen_model, use_gpu=args.gpu)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()