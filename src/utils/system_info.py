import platform
import cpuinfo

def get_device_info():
    """Get CPU and GPU name/model"""
    
    # Get CPU name
    try:
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except:
        cpu_name = platform.processor()
    
    # Get GPU name
    gpu_name = None
    
    # Try NVIDIA GPU (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Try Apple Silicon GPU (MPS)
    if not gpu_name:
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_name = "Apple Silicon GPU"
        except (ImportError, AttributeError):
            pass
    
    # Try TensorFlow
    if not gpu_name:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_name = gpus[0].name
        except ImportError:
            pass
    
    return {
        "cpu": cpu_name,
        "gpu": gpu_name if gpu_name else "No GPU detected"
    }

if __name__ == "__main__":
    devices = get_device_info()
    print(f"CPU: {devices['cpu']}")
    print(f"GPU: {devices['gpu']}")