#!/usr/bin/env python
"""
Enhanced RAG Retriever with GPU optimization and FAISS vector database.
Retrieves most relevant document chunks based on semantic similarity.
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Dict, Tuple, Optional
from rag_encoder import RAGEncoder
from gpu_utils import get_device, clear_gpu_memory
from tqdm import tqdm


class FAISSRetriever:
    """FAISS-based document retriever with GPU optimization"""
    
    def __init__(self, index_path: str, metadata_path: str, chunks_df_path: str,
                 encoder: Optional[RAGEncoder] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            chunks_df_path: Path to chunks CSV file
            encoder: RAGEncoder instance (will create if None)
            device: torch device
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks_df_path = chunks_df_path
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        
        # Load index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load chunks
        self.chunks_df = pd.read_csv(chunks_df_path)
        
        # Initialize encoder
        self.encoder = encoder or RAGEncoder()
    
    def retrieve(self, query: str, k: int = 5, 
                 score_threshold: Optional[float] = None,
                 return_scores: bool = True) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            score_threshold: Only return chunks above this similarity score
            return_scores: Include similarity scores in results
            
        Returns:
            List of retrieved chunks with metadata
        """
        # Encode query
        query_emb = self.encoder.encode_single(query, normalize=True)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_emb, k)
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if idx == -1:  # Invalid index
                break
            
            if score_threshold is not None and score < score_threshold:
                continue
            
            # Get metadata and chunk
            metadata = self.metadata[int(idx)]
            chunk_text = self.chunks_df.iloc[int(idx)]['text']
            
            result = {
                'rank': rank,
                'chunk_id': metadata.get('chunk_id'),
                'source': metadata.get('source'),
                'score': float(score),
                'snippet': chunk_text.replace('\n', ' ')[:500],
                'full_text': chunk_text
            }
            
            results.append(result)
        
        return results
    
    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """
        Retrieve chunks for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of chunks per query
            
        Returns:
            List of result lists (one per query)
        """
        # Encode queries
        query_embs = self.encoder.encode_queries(queries, normalize=True).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embs, k)
        
        all_results = []
        for q_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            results = []
            for rank, (idx, score) in enumerate(zip(query_indices, query_scores), 1):
                if idx == -1:
                    break
                
                metadata = self.metadata[int(idx)]
                chunk_text = self.chunks_df.iloc[int(idx)]['text']
                
                result = {
                    'rank': rank,
                    'chunk_id': metadata.get('chunk_id'),
                    'source': metadata.get('source'),
                    'score': float(score),
                    'snippet': chunk_text.replace('\n', ' ')[:500],
                    'full_text': chunk_text
                }
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def retrieve_with_filters(self, query: str, k: int = 5,
                             source_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve chunks with optional filtering by source.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve (before filtering)
            source_filter: Only include chunks from this source
            
        Returns:
            Filtered list of chunks
        """
        results = self.retrieve(query, k=min(k*3, 50))  # Get more to filter
        
        if source_filter:
            results = [r for r in results if r['source'] == source_filter]
            results = results[:k]
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get retriever statistics"""
        return {
            'total_chunks': self.index.ntotal,
            'embedding_dim': self.index.d,
            'num_unique_sources': len(set(m.get('source') for m in self.metadata)),
            'index_path': self.index_path
        }


class HybridRetriever:
    """Hybrid retriever combining FAISS and keyword search"""
    
    def __init__(self, faiss_retriever: FAISSRetriever, chunks_df: Optional[pd.DataFrame] = None):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_retriever: FAISSRetriever instance
            chunks_df: DataFrame with document chunks (for keyword search)
        """
        self.faiss_retriever = faiss_retriever
        self.chunks_df = chunks_df or faiss_retriever.chunks_df
    
    def retrieve_hybrid(self, query: str, k: int = 5, 
                       faiss_weight: float = 0.7,
                       keyword_weight: float = 0.3) -> List[Dict]:
        """
        Retrieve using combined semantic and keyword search.
        
        Args:
            query: Query string
            k: Number of results
            faiss_weight: Weight for FAISS semantic search
            keyword_weight: Weight for keyword search
            
        Returns:
            Combined and reranked results
        """
        # Get semantic results
        semantic_results = self.faiss_retriever.retrieve(query, k=k*2)
        
        # Get keyword results
        query_terms = set(query.lower().split())
        keyword_scores = []
        
        for idx, row in self.chunks_df.iterrows():
            text = row['text'].lower()
            matches = sum(1 for term in query_terms if term in text)
            score = matches / max(len(query_terms), 1)
            keyword_scores.append((idx, score))
        
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keyword_results = keyword_scores[:k*2]
        
        # Combine results with weighted scoring
        combined = {}
        
        for r in semantic_results:
            chunk_id = r['chunk_id']
            if chunk_id not in combined:
                combined[chunk_id] = r.copy()
                combined[chunk_id]['score'] = r['score'] * faiss_weight
            else:
                combined[chunk_id]['score'] += r['score'] * faiss_weight
        
        for idx, kw_score in keyword_results:
            metadata = self.faiss_retriever.metadata[idx]
            chunk_id = metadata.get('chunk_id')
            
            if chunk_id in combined:
                combined[chunk_id]['score'] += kw_score * keyword_weight
            else:
                chunk_text = self.chunks_df.iloc[idx]['text']
                combined[chunk_id] = {
                    'rank': 0,
                    'chunk_id': chunk_id,
                    'source': metadata.get('source'),
                    'score': kw_score * keyword_weight,
                    'snippet': chunk_text.replace('\n', ' ')[:500],
                    'full_text': chunk_text
                }
        
        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:k]
        
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
        
        return results


def create_faiss_index(embeddings: np.ndarray, 
                      output_index_path: str,
                      use_gpu: bool = True) -> faiss.Index:
    """
    Create FAISS index from embeddings.
    
    Args:
        embeddings: Embedding matrix (num_docs, embedding_dim)
        output_index_path: Path to save index
        use_gpu: Use GPU acceleration
        
    Returns:
        FAISS index
    """
    embeddings = embeddings.astype(np.float32)
    embedding_dim = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatL2(embedding_dim)
    
    if use_gpu and torch.cuda.is_available():
        print("Transferring index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save (CPU version)
    cpu_index = faiss.index_gpu_to_cpu(index) if torch.cuda.is_available() else index
    faiss.write_index(cpu_index, output_index_path)
    print(f"Saved FAISS index to {output_index_path}")
    
    return cpu_index


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Retriever')
    parser.add_argument('--query', required=True, help='Query string')
    parser.add_argument('--index', required=True, help='FAISS index path')
    parser.add_argument('--meta', required=True, help='Metadata JSON path')
    parser.add_argument('--chunks', required=True, help='Chunks CSV path')
    parser.add_argument('--k', type=int, default=5, help='Number of results')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid retrieval')
    
    args = parser.parse_args()
    
    device = get_device(use_gpu=args.gpu) if args.gpu else torch.device('cpu')
    encoder = RAGEncoder(model_name=args.model, device=device)
    retriever = FAISSRetriever(args.index, args.meta, args.chunks, encoder=encoder, device=device)
    
    if args.hybrid:
        hybrid_retriever = HybridRetriever(retriever)
        results = hybrid_retriever.retrieve_hybrid(args.query, k=args.k)
    else:
        results = retriever.retrieve(args.query, k=args.k)
    
    for r in results:
        print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f} | {r['source']}")
        print(f"Snippet: {r['snippet']}")
