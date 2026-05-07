# GPU Setup Guide for RAG System

## Overview

Your current PyTorch installation is CPU-only. To enable GPU acceleration for your RAG system, you need to install PyTorch with CUDA support.

## Current Status

Run this to check current GPU availability:
```bash
python gpu_check.py
```

## Installing PyTorch with CUDA

### Option 1: NVIDIA GPU (CUDA Support)

If you have an NVIDIA GPU, install PyTorch with CUDA:

```bash
# For CUDA 12.1 (recommended for latest GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: AMD GPU (ROCm Support)

For AMD GPUs, use ROCm:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Option 3: Apple Silicon GPU (Metal Support)

For M1/M2 Macs, PyTorch includes native Metal support:

```bash
pip install torch torchvision torchaudio
```

## Verify GPU Installation

After installation, verify CUDA support:

```bash
python gpu_check.py
```

You should see GPU device information if successful.

## Using GPU with RAG Commands

Once GPU is installed, add `--gpu` flag to commands:

### Retrieval with GPU
```bash
python rag_retrieve.py --query "security controls" --gpu --k 5
```

### Generation with GPU
```bash
python rag_generate.py --query "explain incident response" --gpu
```

### Complete RAG Pipeline with GPU
```bash
python rag_pipeline.py --query "What are access control requirements?" --gpu
```

### Interactive RAG with GPU
```bash
python rag_ask.py --gpu
```

### Model Training with GPU
```bash
python train.py --data-path dataset.csv --model lstm --epochs 10 --use-cuda
```

## Performance Tips

1. **Batch Processing**: Use larger batch sizes when running on GPU (memory permitting)
2. **Mixed Precision**: For even faster performance with compatible GPUs, consider using automatic mixed precision
3. **Model Selection**: Smaller models (like gpt2) run faster than larger ones
4. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi` (for NVIDIA GPUs)

## Troubleshooting

### PyTorch reports CUDA not available

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Reinstall PyTorch with correct CUDA version matching your GPU driver

### Out of Memory (OOM) errors

- Reduce batch size
- Use smaller models
- Process queries one at a time instead of batching

### Still slow despite GPU

- Verify GPU is being used: `nvidia-smi` while running
- Check if model actually loaded on GPU
- Consider CPU optimization if GPU memory transfer is bottleneck

## System Requirements

- **NVIDIA GPU**: NVIDIA GPU with compute capability 3.5+
- **NVIDIA Driver**: Latest stable driver for your GPU
- **CUDA Toolkit**: 11.8 or 12.1
- **cuDNN**: Automatically installed with PyTorch CUDA version