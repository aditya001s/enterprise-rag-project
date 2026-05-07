#!/usr/bin/env python
"""
Unified RAG Orchestrator - Complete RAG Pipeline
Integrates retrieval, self-attention analysis, and generation with full GPU support.
"""

import os
import json
import torch
import argparse
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

from rag_encoder import RAGEncoder
from rag_retriever_enhanced import FAISSRetriever, HybridRetriever
from rag_generator_enhanced import RAGGenerator, ContextAnalyzer, RAGPipeline
from gpu_utils import get_device, get_device_summary, clear_gpu_memory


class RAGOrchestrator:
    """
    Unified RAG orchestrator managing the complete pipeline.
    Coordinates retrieval, analysis, and generation with GPU optimization.
    """
    
    def __init__(self, 
                 index_path: str, 
                 metadata_path: str,
                 chunks_path: str,
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 generator_model: str = 'gpt2',
                 classifier_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 use_hybrid_retrieval: bool = False):
        """
        Initialize RAG orchestrator.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            chunks_path: Path to chunks CSV
            embedding_model: Embedding model name
            generator_model: Generator model name
            classifier_path: Path to trained classifier
            device: torch device
            use_hybrid_retrieval: Use hybrid semantic + keyword retrieval
        """
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        self.use_hybrid = use_hybrid_retrieval
        
        print("=" * 60)
        print("Initializing RAG Orchestrator")
        print("=" * 60)
        
        # Initialize encoder
        print("\n[1/4] Loading Encoder...")
        self.encoder = RAGEncoder(model_name=embedding_model, device=self.device)
        
        # Initialize retriever
        print("\n[2/4] Loading Retriever...")
        self.retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            chunks_df_path=chunks_path,
            encoder=self.encoder,
            device=self.device
        )
        
        if use_hybrid_retrieval:
            self.retriever = HybridRetriever(self.retriever)
            print("Using hybrid retrieval (semantic + keyword)")
        
        # Initialize generator
        print("\n[3/4] Loading Generator...")
        self.generator = RAGGenerator(model_name=generator_model, device=self.device)
        
        # Initialize context analyzer
        print("\n[4/4] Loading Context Analyzer...")
        self.analyzer = ContextAnalyzer(classifier_path=classifier_path, device=self.device)
        
        # Create pipeline
        self.pipeline = RAGPipeline(self.generator, self.analyzer)
        
        print("\n" + "=" * 60)
        print("RAG Orchestrator initialized successfully!")
        print("=" * 60)
        self.print_status()
    
    def process_query(self, query: str, k: int = 5, 
                     rerank: bool = True, temperature: float = 0.7,
                     return_stats: bool = True) -> Dict:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            rerank: Rerank chunks by relevance
            temperature: Generation temperature
            return_stats: Include performance statistics
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        timing = {}
        
        # Step 1: Retrieve
        retrieval_start = time.time()
        if self.use_hybrid:
            retrieved = self.retriever.retrieve_hybrid(query, k=k)
        else:
            retrieved = self.retriever.retrieve(query, k=k)
        timing['retrieval_ms'] = (time.time() - retrieval_start) * 1000
        
        if not retrieved:
            return {
                'query': query,
                'error': 'No relevant documents found',
                'status': 'failed',
                'timing_ms': timing if return_stats else None
            }
        
        # Step 2: Generate response
        generation_start = time.time()
        response_data = self.pipeline.generate_response(
            query, 
            retrieved, 
            rerank=rerank,
            temperature=temperature
        )
        timing['generation_ms'] = (time.time() - generation_start) * 1000
        
        timing['total_ms'] = (time.time() - start_time) * 1000
        
        response_data['status'] = 'success'
        if return_stats:
            response_data['timing_ms'] = timing
        
        return response_data
    
    def batch_process(self, queries: List[str], k: int = 5, 
                     return_stats: bool = True) -> List[Dict]:
        """
        Process multiple queries.
        
        Args:
            queries: List of queries
            k: Chunks to retrieve per query
            return_stats: Include performance statistics
            
        Returns:
            List of response dicts
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)}: {query[:60]}...")
            result = self.process_query(query, k=k, return_stats=return_stats)
            results.append(result)
        
        return results
    
    def interactive_session(self, k: int = 5):
        """
        Start interactive query session.
        
        Args:
            k: Chunks to retrieve per query
        """
        print("\n" + "=" * 60)
        print("RAG Interactive Session")
        print("Type 'quit' or 'exit' to end session")
        print("=" * 60 + "\n")
        
        while True:
            try:
                query = input("\n> Query: ").strip()
            except KeyboardInterrupt:
                print("\nSession ended by user")
                break
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            result = self.process_query(query, k=k)
            
            print("\n" + "-" * 60)
            print(f"Response: {result.get('generated_response', 'No response')}")
            print(f"Retrieved {result.get('num_chunks', 0)} relevant documents")
            
            if result.get('timing_ms'):
                timing = result['timing_ms']
                print(f"\nTiming:")
                print(f"  Retrieval: {timing.get('retrieval_ms', 0):.1f}ms")
                print(f"  Generation: {timing.get('generation_ms', 0):.1f}ms")
                print(f"  Total: {timing.get('total_ms', 0):.1f}ms")
            
            print("\nTop retrieved documents:")
            for i, chunk in enumerate(result.get('top_chunks', []), 1):
                print(f"\n  [{i}] {chunk.get('source', 'Unknown')}")
                print(f"      Score: {chunk.get('score', 0):.4f}")
                print(f"      {chunk.get('snippet', '')[:100]}...")
    
    def print_status(self):
        """Print system status and statistics"""
        stats = get_device_summary()
        
        print("\nSystem Status:")
        if stats.get('available'):
            print(f"  Device: {stats.get('device_name')}")
            print(f"  Total Memory: {stats.get('total_memory_gb', 0):.2f} GB")
            print(f"  Allocated: {stats.get('allocated_memory_gb', 0):.2f} GB")
        else:
            print("  Device: CPU (CUDA not available)")
        
        retriever_stats = self.retriever.get_statistics() if hasattr(self.retriever, 'get_statistics') else self.retriever.faiss_retriever.get_statistics()
        print(f"\nRetrieval Index:")
        print(f"  Total chunks: {retriever_stats.get('total_chunks', 0)}")
        print(f"  Embedding dim: {retriever_stats.get('embedding_dim', 0)}")
        print(f"  Unique sources: {retriever_stats.get('num_unique_sources', 0)}")
        
        print(f"\nModels:")
        print(f"  Embedder: {self.encoder.model_name}")
        print(f"  Generator: {self.generator.model_name}")
    
    def cleanup(self):
        """Clean up GPU memory"""
        print("\nCleaning up...")
        self.encoder.clear_cache()
        self.generator.clear_cache()
        clear_gpu_memory()
        print("Cleanup complete")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict] | Dict, output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified RAG Orchestrator - Complete Document Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive session
  python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --interactive --gpu
  
  # Single query
  python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --query "What are the key requirements?" --gpu
  
  # Batch processing
  python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --query-file queries.txt --output results.json --gpu
  
  # With classifier
  python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --query "Query here" --classifier models/best_attn.pt --gpu
        """
    )
    
    # Required arguments
    parser.add_argument('--index', required=True, help='Path to FAISS index')
    parser.add_argument('--meta', required=True, help='Path to metadata JSON')
    parser.add_argument('--chunks', required=True, help='Path to chunks CSV')
    
    # Query input
    parser.add_argument('--query', help='Single query string')
    parser.add_argument('--query-file', help='File with queries (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    # Model configuration
    parser.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model')
    parser.add_argument('--gen-model', default='gpt2', help='Generator model')
    parser.add_argument('--classifier', help='Classifier model path')
    
    # Retrieval options
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid retrieval')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    
    # Generation options
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    
    # Output and execution
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--stats', action='store_true', help='Show timing statistics')
    parser.add_argument('--config', help='Config JSON file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Verify required files
    for path in [args.index, args.meta, args.chunks]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
    
    # Initialize orchestrator
    device = get_device(use_gpu=args.gpu) if args.gpu else torch.device('cpu')
    orchestrator = RAGOrchestrator(
        index_path=args.index,
        metadata_path=args.meta,
        chunks_path=args.chunks,
        embedding_model=args.embed_model,
        generator_model=args.gen_model,
        classifier_path=args.classifier,
        device=device,
        use_hybrid_retrieval=args.hybrid
    )
    
    try:
        # Process queries based on input mode
        if args.interactive:
            orchestrator.interactive_session(k=args.k)
        
        elif args.query:
            result = orchestrator.process_query(
                args.query,
                k=args.k,
                rerank=not args.no_rerank,
                temperature=args.temperature,
                return_stats=args.stats
            )
            print("\n" + "=" * 60)
            print("Query:", result['query'])
            print("-" * 60)
            print("Response:")
            print(result['generated_response'])
            print("=" * 60)
            
            if args.output:
                save_results(result, args.output)
        
        elif args.query_file:
            with open(args.query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = orchestrator.batch_process(
                queries,
                k=args.k,
                return_stats=args.stats
            )
            
            if args.output:
                save_results(results, args.output)
            
            print(f"\nProcessed {len(results)} queries")
        
        else:
            parser.print_help()
    
    finally:
        orchestrator.cleanup()


if __name__ == '__main__':
    main()
