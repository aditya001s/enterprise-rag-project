#!/usr/bin/env python
"""
GPU Check and Status Tool
Tests GPU availability and displays device information
"""
import torch
import argparse
from gpu_utils import get_device_summary

def main():
    parser = argparse.ArgumentParser(description='Check GPU availability and status')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPU CHECK AND STATUS")
    print("=" * 60)
    
    summary = get_device_summary()
    
    if summary.get('available'):
        print("\n✓ CUDA AVAILABLE")
        print(f"  Device Count: {summary['device_count']}")
        print(f"  Current Device: {summary['current_device']}")
        print(f"  Device Name: {summary['device_name']}")
        print(f"  Total Memory: {summary['total_memory_gb']:.2f} GB")
        print(f"  Allocated Memory: {summary['allocated_memory_gb']:.2f} GB")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("\n✗ CUDA NOT AVAILABLE")
        print("  Using CPU instead")
        print(f"  PyTorch Version: {torch.__version__}")
    
    print("\n" + "=" * 60)
    print("GPU USAGE IN RAG SYSTEM")
    print("=" * 60)
    print("\nTo enable GPU acceleration, add --gpu flag to commands:")
    print("\n  python rag_retrieve.py --query 'your query' --gpu")
    print("  python rag_generate.py --query 'your query' --gpu")
    print("  python rag_pipeline.py --query 'your query' --gpu")
    print("  python rag_ask.py --gpu")
    print("\nFor model training:")
    print("  python train.py --data-path dataset.csv --model lstm --use-cuda")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()