#!/usr/bin/env python
"""
Enhanced RAG Encoder with GPU optimization and batch processing.
Handles semantic embeddings for document chunks and queries.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import os
import json
from gpu_utils import get_device, clear_gpu_memory
from tqdm import tqdm


class RAGEncoder:
    """GPU-optimized encoder for RAG pipeline embeddings"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 device: Optional[torch.device] = None,
                 batch_size: int = 32):
        """
        Initialize RAG encoder.
        
        Args:
            model_name: HuggingFace model ID for embeddings
            device: torch device (GPU/CPU). Auto-detect if None
            batch_size: Batch size for encoding operations
        """
        self.model_name = model_name
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        self.batch_size = batch_size
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_queries(self, queries: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode queries to embeddings.
        
        Args:
            queries: List of query strings
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Array of embeddings (num_queries, embedding_dim)
        """
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(queries) > 10
        )
        return embeddings
    
    def encode_documents(self, documents: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode document chunks to embeddings.
        
        Args:
            documents: List of document chunk strings
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Array of embeddings (num_docs, embedding_dim)
        """
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(documents) > 100
        )
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode single text to embedding."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding
    
    def compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_emb: Query embedding (embedding_dim,)
            doc_embs: Document embeddings (num_docs, embedding_dim)
            
        Returns:
            Similarity scores (num_docs,)
        """
        # Assumes normalized embeddings, so dot product = cosine similarity
        similarities = np.dot(doc_embs, query_emb)
        return similarities
    
    def batch_similarity(self, query_embs: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """
        Compute similarities for multiple queries against documents.
        
        Args:
            query_embs: Query embeddings (num_queries, embedding_dim)
            doc_embs: Document embeddings (num_docs, embedding_dim)
            
        Returns:
            Similarity matrix (num_queries, num_docs)
        """
        similarities = np.dot(query_embs, doc_embs.T)
        return similarities
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        clear_gpu_memory()


class EmbeddingCache:
    """Cache for precomputed embeddings"""
    
    def __init__(self, cache_dir: str = './embedding_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_embeddings(self, embeddings: np.ndarray, name: str, metadata: Optional[Dict] = None):
        """Save embeddings to cache"""
        emb_path = os.path.join(self.cache_dir, f'{name}.npy')
        np.save(emb_path, embeddings)
        
        if metadata:
            meta_path = os.path.join(self.cache_dir, f'{name}_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
        
        print(f"Saved embeddings to {emb_path}")
    
    def load_embeddings(self, name: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load embeddings from cache"""
        emb_path = os.path.join(self.cache_dir, f'{name}.npy')
        embeddings = np.load(emb_path)
        
        metadata = None
        meta_path = os.path.join(self.cache_dir, f'{name}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"Loaded embeddings from {emb_path}")
        return embeddings, metadata
    
    def exists(self, name: str) -> bool:
        """Check if embedding exists in cache"""
        return os.path.exists(os.path.join(self.cache_dir, f'{name}.npy'))


def embed_documents_batch(documents: List[str], output_path: str, 
                         model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                         batch_size: int = 32) -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed documents in batches and save embeddings.
    
    Args:
        documents: List of document strings
        output_path: Path to save embeddings .npy file
        model_name: Embedding model name
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    encoder = RAGEncoder(model_name=model_name, batch_size=batch_size)
    
    print(f"Encoding {len(documents)} documents...")
    embeddings = encoder.encode_documents(documents, normalize=True)
    
    # Save embeddings
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")
    
    metadata = [
        {'doc_id': i, 'length': len(doc)} 
        for i, doc in enumerate(documents)
    ]
    
    encoder.clear_cache()
    
    return embeddings, metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Encoder Utility')
    parser.add_argument('--query', help='Encode a query')
    parser.add_argument('--documents', nargs='+', help='Encode multiple documents')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    device = get_device(use_gpu=args.gpu) if args.gpu else torch.device('cpu')
    encoder = RAGEncoder(model_name=args.model, device=device, batch_size=args.batch_size)
    
    if args.query:
        emb = encoder.encode_queries([args.query], normalize=True)
        print(f"Query embedding shape: {emb.shape}")
        print(f"Sample values: {emb[0][:10]}")
    
    if args.documents:
        embs = encoder.encode_documents(args.documents, normalize=True)
        print(f"Document embeddings shape: {embs.shape}")
        
        if len(args.documents) > 1:
            sims = encoder.batch_similarity(embs[:1], embs)
            print(f"Similarities: {sims[0]}")
