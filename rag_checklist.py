#!/usr/bin/env python
"""
RAG Implementation Checklist and Validation Tool
Verifies your setup and provides step-by-step implementation guide
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


IMPLEMENTATION_CHECKLIST = """
╔════════════════════════════════════════════════════════════════════════╗
║           ENTERPRISE RAG - IMPLEMENTATION CHECKLIST                    ║
║                     Production Deployment Guide                        ║
╚════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
PHASE 1: PREREQUISITES (15 minutes)
═══════════════════════════════════════════════════════════════════════════

□ 1.1 System Requirements
    □ Python 3.8 or higher installed
    □ pip/pip3 available
    □ NVIDIA GPU (optional but recommended)
    □ 8GB+ RAM (16GB+ for batch processing)
    □ 50GB+ disk space

□ 1.2 Verify GPU Setup
    Command: python gpu_check.py
    Expected Output:
      ✓ GPU Available
      ✓ Device name shown
      ✓ Memory displayed
    Fallback: Works on CPU (slower)

□ 1.3 Install Dependencies
    Commands:
      pip install --upgrade pip
      pip install -r requirements.txt
    Verify:
      python -c "import torch; print(torch.__version__)"
      python -c "import faiss; print('FAISS OK')"
      python -c "from sentence_transformers import SentenceTransformer; print('ST OK')"

═══════════════════════════════════════════════════════════════════════════
PHASE 2: DOCUMENT PREPARATION (30-60 minutes)
═══════════════════════════════════════════════════════════════════════════

□ 2.1 Extract Document Chunks
    Command: python extract_pdf.py \\
      --pdf your_document.pdf \\
      --chunk-size 500 \\
      --overlap 50
    
    Produces:
      ✓ extracted_chunks.csv (with 'text' column)
      ✓ embedded metadata

□ 2.2 Generate Embeddings
    Command: python embed.py \\
      --chunks extracted_chunks.csv \\
      --model sentence-transformers/all-MiniLM-L6-v2 \\
      --gpu
    
    Produces:
      ✓ embeddings.npy (precomputed vectors)
      ✓ metas.json (metadata)

□ 2.3 Build FAISS Index
    Command: python build_faiss_index.py \\
      --embeddings embeddings.npy \\
      --index faiss_index.index
    
    Produces:
      ✓ faiss_index.index (vector database)
      ✓ faiss_index_meta.json (metadata mapping)

□ 2.4 Verify Preparation
    Check files exist:
      ✓ extracted_chunks.csv (size > 1MB)
      ✓ embeddings.npy (size ~100MB+ depending on docs)
      ✓ faiss_index.index (size ~100MB+)
      ✓ faiss_index_meta.json
      ✓ metas.json

═══════════════════════════════════════════════════════════════════════════
PHASE 3: BASIC TESTING (15 minutes)
═══════════════════════════════════════════════════════════════════════════

□ 3.1 Test Single Query
    Command: python rag_orchestrator.py \\
      --index faiss_index.index \\
      --meta faiss_index_meta.json \\
      --chunks extracted_chunks.csv \\
      --query "What are the key requirements?" \\
      --gpu
    
    Verify:
      ✓ Query processed successfully
      ✓ Response generated
      ✓ Retrieved chunks shown
      ✓ Performance timing displayed

□ 3.2 Test with Classifier
    Command: python rag_orchestrator.py \\
      --index faiss_index.index \\
      --meta faiss_index_meta.json \\
      --chunks extracted_chunks.csv \\
      --query "Test query" \\
      --classifier models/best_attn.pt \\
      --gpu
    
    Verify:
      ✓ Self-Attention model loaded
      ✓ Chunks ranked by relevance
      ✓ Response generated

□ 3.3 Test Batch Processing
    Create queries.txt:
      What is the main topic?
      What are requirements?
      How does it work?
    
    Command: python rag_orchestrator.py \\
      --index faiss_index.index \\
      --meta faiss_index_meta.json \\
      --chunks extracted_chunks.csv \\
      --query-file queries.txt \\
      --output test_results.json \\
      --gpu
    
    Verify:
      ✓ All queries processed
      ✓ results.json created
      ✓ Results contain responses

═══════════════════════════════════════════════════════════════════════════
PHASE 4: INTERACTIVE TESTING (10 minutes)
═══════════════════════════════════════════════════════════════════════════

□ 4.1 Start Interactive Mode
    Command: python rag_query_tool.py \\
      --index faiss_index.index \\
      --meta faiss_index_meta.json \\
      --chunks extracted_chunks.csv \\
      --gpu
    
    Test queries:
      • What are the main topics?
      • Explain the key concepts
      • How do I use this?
    
    Commands to try:
      • help (show commands)
      • status (show system status)
      • history (show query history)
      • export results.json (save history)
      • quit (exit)

□ 4.2 Verify Performance
    Monitor during interactive mode:
      ✓ Response time < 5 seconds
      ✓ Quality of responses (relevant to queries)
      ✓ Retrieved chunks are appropriate
      ✓ GPU memory stays stable (with --gpu)

═══════════════════════════════════════════════════════════════════════════
PHASE 5: CONFIGURATION & OPTIMIZATION (20 minutes)
═══════════════════════════════════════════════════════════════════════════

□ 5.1 Create Configuration File
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

□ 5.2 Test with Configuration
    Command: python rag_orchestrator.py \\
      --config rag_config.json \\
      --query "Test query" \\
      --gpu

□ 5.3 Optimize Parameters
    Experiment with:
      □ k values: 3, 5, 10 (chunk count)
      □ temperature: 0.3, 0.7, 0.9
      □ Models: distilgpt2 vs gpt2
      □ Hybrid: with --hybrid flag
    
    Measure:
      □ Response quality
      □ Processing time
      □ Relevance of chunks

□ 5.4 Performance Tuning
    For faster responses:
      • Enable GPU: --gpu
      • Reduce k: --k 3
      • Smaller model: --gen-model distilgpt2
      • Increase batch size (if GPU memory allows)
    
    For better quality:
      • Increase k: --k 10
      • Use hybrid: --hybrid
      • Add classifier: --classifier models/best_attn.pt
      • Longer context: increase max_context_tokens

═══════════════════════════════════════════════════════════════════════════
PHASE 6: PRODUCTION DEPLOYMENT (30+ minutes)
═══════════════════════════════════════════════════════════════════════════

□ 6.1 Production Testing
    □ Test with diverse queries (50+)
    □ Validate response quality
    □ Monitor performance metrics
    □ Stress test with batch processing
    □ Test error handling

□ 6.2 Integration Testing
    □ Test in your application
    □ Verify import paths
    □ Test with real data
    □ Monitor memory usage
    □ Check GPU utilization

□ 6.3 Deployment Preparation
    □ Create deployment script
    □ Document configuration
    □ Set up logging
    □ Create backup strategy
    □ Plan maintenance schedule

□ 6.4 User Documentation
    □ Document how to run
    □ Provide example queries
    □ Create troubleshooting guide
    □ Document support channels

□ 6.5 Monitoring & Maintenance
    □ Set up performance monitoring
    □ Track response quality
    □ Monitor resource usage
    □ Plan updates
    □ Create rollback procedure

═══════════════════════════════════════════════════════════════════════════
PHASE 7: VALIDATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════

□ 7.1 Core Functionality
    □ Query processing works
    □ Responses are accurate
    □ Documents retrieved correctly
    □ Performance acceptable

□ 7.2 GPU/Performance
    □ GPU detection works
    □ GPU acceleration speeds up processing
    □ Memory usage is stable
    □ No CUDA errors

□ 7.3 Error Handling
    □ Handles missing files
    □ Handles invalid queries
    □ Handles GPU memory errors
    □ Provides helpful error messages

□ 7.4 Integration
    □ Works with existing models
    □ Self-Attention model preserved
    □ No breaking changes
    □ Backward compatible

□ 7.5 Documentation
    □ README is complete
    □ Examples are working
    □ Troubleshooting guide available
    □ API documentation clear

═══════════════════════════════════════════════════════════════════════════
PHASE 8: QUALITY ASSURANCE
═══════════════════════════════════════════════════════════════════════════

□ 8.1 Response Quality
    Test query categories:
      □ Simple factual questions
      □ Complex multi-part questions
      □ Questions requiring synthesis
      □ Off-topic questions (should say N/A)
    
    Acceptance criteria:
      □ Relevant responses > 80%
      □ Hallucination rate < 5%
      □ Response time < 5s with GPU

□ 8.2 Performance Benchmarks
    Measure and document:
      □ Average response time
      □ GPU memory usage
      □ CPU usage
      □ Throughput (queries/second)
    
    Set alerts for:
      □ Response time > 10s
      □ GPU memory > 90%
      □ Error rate > 1%

□ 8.3 Load Testing
    Test scenarios:
      □ Single user (current state)
      □ 10 concurrent queries
      □ 100 queries in batch
      □ Mixed workload
    
    Success criteria:
      □ No crashes
      □ Performance degradation < 50%
      □ Memory stable

═══════════════════════════════════════════════════════════════════════════
FINAL VERIFICATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════

✅ READY FOR PRODUCTION if all of the following are checked:

  ✓ All files present and validated
  ✓ GPU working correctly (or CPU fallback tested)
  ✓ Document preparation complete
  ✓ All tests passing
  ✓ Performance acceptable
  ✓ Configuration optimized
  ✓ Documentation complete
  ✓ Error handling tested
  ✓ User training completed
  ✓ Support plan in place

═══════════════════════════════════════════════════════════════════════════
QUICK REFERENCE - COMMON COMMANDS
═══════════════════════════════════════════════════════════════════════════

# Check setup
python gpu_check.py

# Interactive mode
python rag_query_tool.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --gpu

# Single query
python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --query "Question" --gpu

# Batch processing
python rag_orchestrator.py --index faiss_index.index --meta faiss_index_meta.json --chunks extracted_chunks.csv --query-file queries.txt --output results.json --gpu

# With configuration
python rag_orchestrator.py --config rag_config.json --query "Question" --gpu

# Get help
python rag_quickstart.py guide
python rag_quickstart.py troubleshooting

═══════════════════════════════════════════════════════════════════════════
ESTIMATED TIMELINE
═══════════════════════════════════════════════════════════════════════════

Total Implementation Time: 2-4 hours

  Phase 1 (Prerequisites):     15 min
  Phase 2 (Document Prep):     30-60 min (depends on document size)
  Phase 3 (Basic Testing):     15 min
  Phase 4 (Interactive):       10 min
  Phase 5 (Configuration):     20 min
  Phase 6 (Production):        30+ min (depends on complexity)
  Phase 7 (Validation):        20 min
  Phase 8 (QA):               30 min

═══════════════════════════════════════════════════════════════════════════
"""


VALIDATION_SCRIPT = """
#!/usr/bin/env python
\"\"\"Automated validation script\"\"\"

import os
import sys
import json
from pathlib import Path

class RAGValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks = []
    
    def check_files(self):
        \"\"\"Check required files exist\"\"\"
        required_files = [
            'extracted_chunks.csv',
            'faiss_index.index',
            'faiss_index_meta.json',
            'metas.json',
            'embeddings.npy'
        ]
        
        for f in required_files:
            if os.path.exists(f):
                size = os.path.getsize(f) / (1024*1024)  # MB
                self.checks.append(f'✓ {f} ({size:.1f}MB)')
            else:
                self.errors.append(f'✗ Missing: {f}')
    
    def check_models(self):
        \"\"\"Check model files\"\"\"
        models = [
            'models/best_rnn.pt',
            'models/best_lstm.pt',
            'models/best_attn.pt'
        ]
        
        for m in models:
            if os.path.exists(m):
                self.checks.append(f'✓ {m}')
            else:
                self.warnings.append(f'⚠ Optional: {m}')
    
    def check_imports(self):
        \"\"\"Check required packages\"\"\"
        try:
            import torch
            self.checks.append(f'✓ PyTorch {torch.__version__}')
        except:
            self.errors.append('✗ PyTorch not installed')
        
        try:
            import faiss
            self.checks.append('✓ FAISS installed')
        except:
            self.errors.append('✗ FAISS not installed')
        
        try:
            from sentence_transformers import SentenceTransformer
            self.checks.append('✓ Sentence-Transformers installed')
        except:
            self.errors.append('✗ Sentence-Transformers not installed')
    
    def check_gpu(self):
        \"\"\"Check GPU availability\"\"\"
        try:
            import torch
            if torch.cuda.is_available():
                self.checks.append(f'✓ GPU: {torch.cuda.get_device_name(0)}')
            else:
                self.warnings.append('⚠ GPU not available (CPU mode)')
        except:
            pass
    
    def report(self):
        \"\"\"Print validation report\"\"\"
        print('\\n' + '='*60)
        print('RAG VALIDATION REPORT')
        print('='*60)
        
        print('\\nPASSED CHECKS:')
        for c in self.checks:
            print(f'  {c}')
        
        if self.warnings:
            print('\\nWARNINGS:')
            for w in self.warnings:
                print(f'  {w}')
        
        if self.errors:
            print('\\nERRORS:')
            for e in self.errors:
                print(f'  {e}')
            return False
        
        print('\\n✓ System ready for production!')
        return True
    
    def validate_all(self):
        self.check_imports()
        self.check_gpu()
        self.check_files()
        self.check_models()
        return self.report()

if __name__ == '__main__':
    validator = RAGValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)
"""


def main():
    parser = argparse.ArgumentParser(
        description='RAG Implementation Checklist'
    )
    parser.add_argument('--print-checklist', action='store_true')
    parser.add_argument('--save-validation', type=str, help='Save validation script')
    parser.add_argument('--run-validation', action='store_true')
    
    args = parser.parse_args()
    
    if args.print_checklist or not args.save_validation and not args.run_validation:
        print(IMPLEMENTATION_CHECKLIST)
    
    if args.save_validation:
        with open(args.save_validation, 'w') as f:
            f.write(VALIDATION_SCRIPT)
        print(f"✓ Validation script saved to {args.save_validation}")


if __name__ == '__main__':
    main()
