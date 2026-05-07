import torch

def get_device(use_gpu=True):
    """Get the appropriate device (GPU or CPU)"""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device_summary():
    """Get GPU device summary"""
    if torch.cuda.is_available():
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0),
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'allocated_memory_gb': torch.cuda.memory_allocated() / 1e9
        }
    else:
        return {'available': False, 'message': 'CUDA not available'}
