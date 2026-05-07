#!/usr/bin/env python
"""
RAG Quick Start Guide and Example Runner
Demonstrates how to use the complete RAG pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path


QUICK_START_GUIDE = """
╔════════════════════════════════════════════════════════════════════════╗
║                    ENTERPRISE RAG - QUICK START GUIDE                  ║
╚════════════════════════════════════════════════════════════════════════╝

📋 WHAT IS RAG?
Retrieval-Augmented Generation combines semantic search with language 
models to answer questions grounded in your documents.

✨ KEY BENEFITS:
  • Reduces hallucination - Answers backed by your documents
  • Improves accuracy - Uses real enterprise knowledge
  • GPU accelerated - 10-15× faster with GPU
  • Self-Attention integrated - Leverages your trained models

════════════════════════════════════════════════════════════════════════════

⚡ 5-MINUTE QUICK START

Step 1: Check GPU Status
────────────────────────
  python gpu_check.py
  
  This shows:
    ✓ GPU availability
    ✓ CUDA version
    ✓ GPU memory

Step 2: Start Interactive Query (if you have documents prepared)
────────────────────────────────────────────────────────────────
  python rag_query_tool.py \\
    --index faiss_index.index \\
    --meta faiss_index_meta.json \\
    --chunks extracted_chunks.csv \\
    --gpu

  Try asking:
    • What are the system requirements?
    • Explain the security policies
    • What compliance standards apply?

Step 3: Single Query Example
────────────────────────────
  python rag_orchestrator.py \\
    --index faiss_index.index \\
    --meta faiss_index_meta.json \\
    --chunks extracted_chunks.csv \\
    --query "What are the access controls?" \\
    --k 5 \\
    --gpu

Step 4: With Your Self-Attention Model
───────────────────────────────────────
  python rag_query_tool.py \\
    --index faiss_index.index \\
    --meta faiss_index_meta.json \\
    --chunks extracted_chunks.csv \\
    --classifier models/best_attn.pt \\
    --gpu

════════════════════════════════════════════════════════════════════════════

📚 FULL DOCUMENT PREPARATION PIPELINE

If you're starting with PDF documents:

1. Extract Document Chunks
   ────────────────────────
   python extract_pdf.py \\
     --pdf path/to/document.pdf \\
     --chunk-size 500 \\
     --overlap 50

2. Generate Embeddings
   ───────────────────
   python embed.py \\
     --chunks extracted_chunks.csv \\
     --model sentence-transformers/all-MiniLM-L6-v2

3. Build FAISS Index
   ──────────────────
   python build_faiss_index.py \\
     --embeddings embeddings.npy \\
     --index faiss_index.index

4. Start Using RAG!
   ────────────────
   python rag_query_tool.py \\
     --index faiss_index.index \\
     --meta faiss_index_meta.json \\
     --chunks extracted_chunks.csv \\
     --gpu

════════════════════════════════════════════════════════════════════════════

🎯 DIFFERENT USE CASES

Use Case 1: Customer Support Q&A
────────────────────────────────
python rag_query_tool.py \\
  --index faiss_index.index \\
  --meta faiss_index_meta.json \\
  --chunks extracted_chunks.csv \\
  --gen-model gpt2 \\
  --k 3 \\
  --gpu

Use Case 2: Policy Compliance Check
────────────────────────────────────
python rag_orchestrator.py \\
  --index faiss_index.index \\
  --meta faiss_index_meta.json \\
  --chunks extracted_chunks.csv \\
  --query "Is this design compliant with our policies?" \\
  --classifier models/best_attn.pt \\
  --hybrid \\
  --gpu

Use Case 3: Batch Document Analysis
────────────────────────────────────
# Create queries.txt with questions
python rag_orchestrator.py \\
  --index faiss_index.index \\
  --meta faiss_index_meta.json \\
  --chunks extracted_chunks.csv \\
  --query-file queries.txt \\
  --output analysis_results.json \\
  --gpu

Use Case 4: Integration in Application
───────────────────────────────────────
from rag_orchestrator import RAGOrchestrator
import torch

orchestrator = RAGOrchestrator(
    index_path='faiss_index.index',
    metadata_path='faiss_index_meta.json',
    chunks_path='extracted_chunks.csv',
    device=torch.device('cuda')
)

result = orchestrator.process_query(
    "Your question here",
    k=5,
    return_stats=True
)

print(result['generated_response'])

════════════════════════════════════════════════════════════════════════════

⚙️ CONFIGURATION

Create rag_config.json:
{
  "index": "faiss_index.index",
  "meta": "faiss_index_meta.json",
  "chunks": "extracted_chunks.csv",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "gen_model": "gpt2",
  "classifier": "models/best_attn.pt",
  "k": 5,
  "temperature": 0.7,
  "hybrid": true
}

Use it:
python rag_orchestrator.py --config rag_config.json --query "?" --gpu

════════════════════════════════════════════════════════════════════════════

📊 PERFORMANCE TIPS

1. GPU Acceleration (Required!)
   • Add --gpu to all commands
   • 10-15× faster performance
   • Minimum: 6GB VRAM

2. Batch Size Optimization
   • Default: 32 (good balance)
   • Increase to 64 for speed (with 12GB+ VRAM)
   • Decrease to 16 for memory constraints

3. Model Selection
   • Embedder: all-MiniLM-L6-v2 (fast, good quality)
   • Generator: gpt2 (fast), distilgpt2 (faster)
   • Use your best_attn.pt for classification

4. Retrieval Parameters
   • k=3 for fast interactive queries
   • k=10 for comprehensive analysis
   • Use --hybrid for technical docs

════════════════════════════════════════════════════════════════════════════

❓ TROUBLESHOOTING

Problem: "FAISS index not found"
└─ Solution: Run document preparation steps (extract → embed → build)

Problem: Out of GPU memory
└─ Solution: Use CPU mode or reduce batch size
         python rag_query_tool.py ... (without --gpu)

Problem: Slow response generation
└─ Solution: Use distilgpt2, enable GPU, reduce context

Problem: Inaccurate answers
└─ Solution: Try --hybrid retrieval, increase k, improve document chunks

════════════════════════════════════════════════════════════════════════════

📖 ARCHITECTURE OVERVIEW

User Query
    ↓
[Encoder] - Convert query to embeddings (GPU)
    ↓
[Retriever] - Find top-K relevant documents (GPU)
    ↓
[Analyzer] - Rank by relevance using self-attention
    ↓
[Generator] - Create response grounded in retrieved documents (GPU)
    ↓
Final Answer (backed by your documents!)

════════════════════════════════════════════════════════════════════════════

✅ YOUR MODELS ARE SAFE

Your existing trained models remain UNCHANGED:
  ✓ models/best_rnn.pt  - RNN model
  ✓ models/best_lstm.pt - LSTM model
  ✓ models/best_attn.pt - Self-Attention model

The RAG system USES but doesn't MODIFY these models.

════════════════════════════════════════════════════════════════════════════

🚀 NEXT STEPS

1. Verify GPU: python gpu_check.py
2. Prepare documents (if needed): python extract_pdf.py ...
3. Start interactive mode: python rag_query_tool.py ...
4. Try different queries and parameters
5. Integrate into your application

════════════════════════════════════════════════════════════════════════════

💡 EXAMPLE QUERIES

• What are the system requirements?
• Explain the security policies in detail
• What compliance standards apply?
• How do I configure the system?
• What are the troubleshooting steps?
• Summarize the access control mechanisms
• What authentication methods are supported?

════════════════════════════════════════════════════════════════════════════
"""


EXAMPLE_CODE = """
#!/usr/bin/env python
\"\"\"Example: Using RAG in Python\"\"\"

import torch
from rag_orchestrator import RAGOrchestrator
from gpu_utils import get_device

# Initialize orchestrator
device = get_device(use_gpu=True)

orchestrator = RAGOrchestrator(
    index_path='faiss_index.index',
    metadata_path='faiss_index_meta.json',
    chunks_path='extracted_chunks.csv',
    device=device,
    use_hybrid_retrieval=True
)

# Single query
query = "What are the system requirements?"
result = orchestrator.process_query(query, k=5, return_stats=True)

print(f"Query: {result['query']}")
print(f"Response: {result['generated_response']}")
print(f"Processing time: {result['timing_ms']['total_ms']:.0f}ms")

# Batch queries
queries = [
    "What are the security requirements?",
    "How do I configure the system?",
    "What backup strategies are recommended?"
]

results = orchestrator.batch_process(queries)

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Cleanup
orchestrator.cleanup()
"""


EXAMPLE_CONFIG = """
{
  "index": "faiss_index.index",
  "meta": "faiss_index_meta.json",
  "chunks": "extracted_chunks.csv",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "gen_model": "gpt2",
  "classifier": "models/best_attn.pt",
  "k": 5,
  "temperature": 0.7,
  "hybrid": true,
  "description": "Enterprise RAG Configuration"
}
"""


TROUBLESHOOTING_GUIDE = """
ENTERPRISE RAG - TROUBLESHOOTING GUIDE
═════════════════════════════════════

1. GPU/CUDA Issues
──────────────────
Problem: "CUDA out of memory"
Solutions:
  a) Use CPU mode: python rag_query_tool.py ... (without --gpu)
  b) Reduce batch size in code
  c) Clear GPU cache between queries

Check Status: python gpu_check.py

2. Index/Data Issues
────────────────────
Problem: "FAISS index not found"
Solution: Build the index first
  python extract_pdf.py --pdf document.pdf
  python embed.py --chunks extracted_chunks.csv
  python build_faiss_index.py --embeddings embeddings.npy

3. Model Issues
───────────────
Problem: "Model not found" or "Permission denied"
Solution: 
  1) Check file paths are correct
  2) Ensure files have read permissions
  3) Verify model directory structure

4. Slow Performance
────────────────────
Problem: Response takes >5 seconds
Solutions:
  1) Enable GPU: add --gpu flag
  2) Use smaller model: --gen-model distilgpt2
  3) Reduce k (chunks): --k 3
  4) Check GPU utilization: python gpu_check.py

5. Poor Answer Quality
───────────────────────
Problem: Answers seem generic or off-topic
Solutions:
  1) Increase k (more chunks): --k 10
  2) Use hybrid retrieval: --hybrid
  3) Use classifier for reranking: --classifier models/best_attn.pt
  4) Check document chunks are well-formatted

6. Installation Issues
───────────────────────
Problem: "ImportError: No module named ..."
Solution: Reinstall requirements
  pip install --upgrade -r requirements.txt

Common issues:
  • sentence-transformers: pip install sentence-transformers
  • faiss: pip install faiss-cpu
  • transformers: pip install transformers

7. Batch Processing Issues
─────────────────────────
Problem: "Incomplete results" or "Process crashed"
Solutions:
  1) Reduce batch size
  2) Use GPU for speed
  3) Split into smaller batches
  4) Check queries.txt format (one per line)

8. Integration Issues
─────────────────────
Problem: "Module import failed in custom code"
Solution: Ensure PYTHONPATH includes training directory
  export PYTHONPATH=$PYTHONPATH:/path/to/training
  python your_script.py

═════════════════════════════════════════════════════════════

DEBUG MODE
──────────
Run with verbose output:
  python rag_query_tool.py ... --debug

Check individual components:
  # Test encoder
  python rag_encoder.py --query "test" --gpu
  
  # Test retriever
  python rag_retriever_enhanced.py --query "test" --index ... --gpu
  
  # Test generator
  python rag_generator_enhanced.py --query "test" --model gpt2 --gpu

═════════════════════════════════════════════════════════════
"""


def main():
    parser = argparse.ArgumentParser(
        description='RAG Quick Start Guide',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('action', nargs='?', default='guide',
                       choices=['guide', 'example', 'config', 'troubleshooting', 'all'],
                       help='What to display')
    
    parser.add_argument('--save-example', type=str, help='Save example code to file')
    parser.add_argument('--save-config', type=str, help='Save example config to file')
    
    args = parser.parse_args()
    
    if args.action == 'guide' or args.action == 'all':
        print(QUICK_START_GUIDE)
    
    if args.action == 'example' or args.action == 'all':
        print("\n" + "="*80)
        print("PYTHON EXAMPLE CODE")
        print("="*80 + "\n")
        print(EXAMPLE_CODE)
        
        if args.save_example:
            with open(args.save_example, 'w') as f:
                f.write(EXAMPLE_CODE)
            print(f"\n✓ Saved to {args.save_example}")
    
    if args.action == 'config' or args.action == 'all':
        print("\n" + "="*80)
        print("EXAMPLE CONFIGURATION")
        print("="*80 + "\n")
        print(EXAMPLE_CONFIG)
        
        if args.save_config:
            with open(args.save_config, 'w') as f:
                f.write(EXAMPLE_CONFIG)
            print(f"\n✓ Saved to {args.save_config}")
    
    if args.action == 'troubleshooting' or args.action == 'all':
        print("\n" + "="*80)
        print("TROUBLESHOOTING GUIDE")
        print("="*80 + "\n")
        print(TROUBLESHOOTING_GUIDE)


if __name__ == '__main__':
    main()
