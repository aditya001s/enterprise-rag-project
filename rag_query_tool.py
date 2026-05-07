#!/usr/bin/env python
"""
Enhanced Interactive RAG Query Tool
User-friendly interface for querying enterprise documents with RAG pipeline.
"""

import os
import sys
import json
import time
import argparse
import torch
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from rag_orchestrator import RAGOrchestrator
from gpu_utils import get_device_summary


class RAGQueryTool:
    """Interactive RAG query tool with enhanced UX"""
    
    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator
        self.query_history = []
        self.start_time = time.time()
    
    def print_header(self):
        """Print application header"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  ENTERPRISE DOCUMENT ANALYSIS WITH RAG".center(78) + "║")
        print("║" + "  Retrieval-Augmented Generation Pipeline".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
    
    def print_footer(self):
        """Print application footer"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        uptime = time.time() - self.start_time
        uptime_str = f"Session uptime: {int(uptime)}s | Queries processed: {len(self.query_history)}"
        print("║" + uptime_str.ljust(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
    
    def print_help(self):
        """Print help menu"""
        print("\n" + "─" * 80)
        print("AVAILABLE COMMANDS:".ljust(80))
        print("─" * 80)
        print("""
  help              Show this help menu
  status            Show system and index status
  history           Show query history
  export [file]     Export query history to JSON
  clear             Clear query history
  quit / exit       Exit the application
  
QUERY OPTIONS:
  Type your question and press Enter to search the document database.
  Examples:
    - What are the access control requirements?
    - Explain the security policies
    - Show me compliance requirements
""")
        print("─" * 80 + "\n")
    
    def print_status(self):
        """Print detailed system status"""
        print("\n" + "─" * 80)
        print("SYSTEM STATUS".ljust(80))
        print("─" * 80)
        
        # GPU/Device info
        gpu_stats = get_device_summary()
        if gpu_stats.get('available'):
            print(f"\n📊 Device Information:")
            print(f"   Device:           {gpu_stats.get('device_name', 'N/A')}")
            print(f"   Total Memory:     {gpu_stats.get('total_memory_gb', 0):.2f} GB")
            print(f"   Allocated:        {gpu_stats.get('allocated_memory_gb', 0):.2f} GB")
            print(f"   Device Count:     {gpu_stats.get('device_count', 1)}")
        else:
            print(f"\n📊 Device Information:")
            print(f"   Status:           CPU Mode (CUDA not available)")
        
        # Index stats
        try:
            if hasattr(self.orchestrator.retriever, 'get_statistics'):
                stats = self.orchestrator.retriever.get_statistics()
            else:
                stats = self.orchestrator.retriever.faiss_retriever.get_statistics()
            
            print(f"\n📚 Index Information:")
            print(f"   Total Chunks:     {stats.get('total_chunks', 0):,}")
            print(f"   Embedding Dim:    {stats.get('embedding_dim', 0)}")
            print(f"   Unique Sources:   {stats.get('num_unique_sources', 0)}")
        except Exception as e:
            print(f"\n📚 Index Information: Error retrieving stats - {e}")
        
        # Models
        print(f"\n🤖 Models:")
        print(f"   Encoder:          {self.orchestrator.encoder.model_name}")
        print(f"   Generator:        {self.orchestrator.generator.model_name}")
        
        # Session stats
        print(f"\n⏱️  Session Statistics:")
        uptime = time.time() - self.start_time
        print(f"   Uptime:           {int(uptime)}s")
        print(f"   Queries:          {len(self.query_history)}")
        
        print("\n" + "─" * 80 + "\n")
    
    def print_response(self, result: Dict, show_chunks: bool = True):
        """Print formatted response"""
        print("\n" + "┌" + "─" * 78 + "┐")
        
        # Response
        print("│ Response:".ljust(79) + "│")
        print("├" + "─" * 78 + "┤")
        
        response_text = result.get('generated_response', 'No response generated')
        words = response_text.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 76:
                print("│ " + line.ljust(76) + " │")
                line = word
            else:
                line += (" " if line else "") + word
        
        if line:
            print("│ " + line.ljust(76) + " │")
        
        print("├" + "─" * 78 + "┤")
        
        # Metadata
        timing = result.get('timing_ms', {})
        num_chunks = result.get('num_chunks', 0)
        
        timing_str = f"Retrieval: {timing.get('retrieval_ms', 0):.0f}ms | Gen: {timing.get('generation_ms', 0):.0f}ms | Total: {timing.get('total_ms', 0):.0f}ms"
        print(f"│ {('Chunks: ' + str(num_chunks)).ljust(20)} {timing_str.rjust(56)} │")
        
        if show_chunks and result.get('top_chunks'):
            print("├" + "─" * 78 + "┤")
            print("│ Top Retrieved Documents:".ljust(79) + "│")
            
            for i, chunk in enumerate(result['top_chunks'], 1):
                source = chunk.get('source', 'Unknown')[:40]
                score = chunk.get('score', 0)
                print(f"│   [{i}] {source.ljust(40)} (Score: {score:.4f})".ljust(79) + "│")
                
                snippet = chunk.get('snippet', '')[:70]
                print(f"│       {snippet}...".ljust(79) + "│")
        
        print("└" + "─" * 78 + "┘\n")
    
    def run_interactive(self, k: int = 5, show_help_on_start: bool = True):
        """Run interactive session"""
        self.print_header()
        
        if show_help_on_start:
            self.print_help()
        
        print("Type 'help' for available commands or start typing your query.\n")
        
        while True:
            try:
                user_input = input("🔍 Query: ").strip()
            except KeyboardInterrupt:
                print("\n\nSession interrupted by user")
                break
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using Enterprise RAG. Goodbye! 👋")
                break
            
            elif user_input.lower() == 'help':
                self.print_help()
                continue
            
            elif user_input.lower() == 'status':
                self.print_status()
                continue
            
            elif user_input.lower() == 'history':
                self.print_history()
                continue
            
            elif user_input.lower().startswith('export'):
                parts = user_input.split()
                filename = parts[1] if len(parts) > 1 else 'rag_history.json'
                self.export_history(filename)
                continue
            
            elif user_input.lower() == 'clear':
                self.query_history = []
                print("✓ Query history cleared")
                continue
            
            # Process query
            print("\n⏳ Processing query...\n")
            
            try:
                result = self.orchestrator.process_query(
                    user_input,
                    k=k,
                    return_stats=True
                )
                
                if result.get('status') == 'success':
                    self.print_response(result)
                    self.query_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'query': user_input,
                        'result': result
                    })
                else:
                    print(f"⚠️  Error: {result.get('error', 'Unknown error')}\n")
            
            except Exception as e:
                print(f"❌ Error processing query: {e}\n")
        
        self.print_footer()
    
    def print_history(self):
        """Print query history"""
        if not self.query_history:
            print("\nNo queries in history\n")
            return
        
        print("\n" + "─" * 80)
        print("QUERY HISTORY".ljust(80))
        print("─" * 80)
        
        for i, entry in enumerate(self.query_history, 1):
            timestamp = entry.get('timestamp', 'N/A')
            query = entry.get('query', '')[:60]
            print(f"\n[{i}] {timestamp}")
            print(f"    Query: {query}")
            
            result = entry.get('result', {})
            if result:
                timing = result.get('timing_ms', {})
                print(f"    Time: {timing.get('total_ms', 0):.0f}ms | Chunks: {result.get('num_chunks', 0)}")
        
        print("\n" + "─" * 80 + "\n")
    
    def export_history(self, filename: str):
        """Export query history to JSON"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.query_history, f, indent=2)
        
        print(f"✓ History exported to {filename}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive RAG Query Tool for Enterprise Documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
  python rag_query_tool.py --index faiss_index.index --meta faiss_index_meta.json \\
    --chunks extracted_chunks.csv --gpu

GPU Acceleration:
  Add --gpu flag to enable CUDA acceleration for faster queries

Configuration:
  python rag_query_tool.py --config rag_config.json --interactive --gpu
        """
    )
    
    # Required
    parser.add_argument('--index', required=True, help='Path to FAISS index')
    parser.add_argument('--meta', required=True, help='Path to metadata JSON')
    parser.add_argument('--chunks', required=True, help='Path to chunks CSV')
    
    # Options
    parser.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--gen-model', default='gpt2')
    parser.add_argument('--classifier', help='Classifier model path')
    parser.add_argument('--k', type=int, default=5, help='Chunks to retrieve')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid retrieval')
    parser.add_argument('--config', help='Config JSON file')
    
    args = parser.parse_args()
    
    # Verify files
    for path in [args.index, args.meta, args.chunks]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
    
    # Initialize
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    from gpu_utils import get_device
    device = get_device(use_gpu=args.gpu)
    
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
    
    tool = RAGQueryTool(orchestrator)
    
    try:
        tool.run_interactive(k=args.k)
    finally:
        orchestrator.cleanup()


if __name__ == '__main__':
    main()
