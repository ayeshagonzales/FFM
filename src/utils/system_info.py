import platform
import psutil
import cpuinfo
from datetime import datetime

def get_system_info():
    """Collect comprehensive system information for runtime monitoring"""
    
    info = {
        # Basic system info
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        
        # CPU details
        "cpu_name": cpuinfo.get_cpu_info()['brand_raw'],
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        
        # Memory info
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "ram_used_percent": psutil.virtual_memory().percent,
        
        # GPU info (requires additional setup)
        "gpu_available": check_gpu(),
        
        # Python environment
        "python_version": platform.python_version(),
    }
    
    return info

def check_gpu():
    """Check for GPU availability across different frameworks"""
    gpu_info = {
        "has_gpu": False,
        "gpu_type": None,
        "gpu_name": None,
        "gpu_count": 0
    }
    
    # Check for NVIDIA GPU (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "NVIDIA CUDA"
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_count"] = torch.cuda.device_count()
            return gpu_info
    except ImportError:
        pass
    
    # Check for Apple Silicon GPU (MPS)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "Apple MPS"
            gpu_info["gpu_name"] = "Apple Silicon GPU"
            return gpu_info
    except (ImportError, AttributeError):
        pass
    
    # Check via TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "TensorFlow GPU"
            gpu_info["gpu_count"] = len(gpus)
            gpu_info["gpu_name"] = gpus[0].name if gpus else None
            return gpu_info
    except ImportError:
        pass
    
    return gpu_info

def print_system_info(info):
    """Pretty print system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"\n📅 Timestamp: {info['timestamp']}")
    print(f"\n💻 PLATFORM")
    print(f"   OS: {info['platform']} {info['platform_release']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Hostname: {info['hostname']}")
    
    print(f"\n🔧 CPU")
    print(f"   Name: {info['cpu_name']}")
    print(f"   Physical Cores: {info['cpu_cores_physical']}")
    print(f"   Logical Cores: {info['cpu_cores_logical']}")
    print(f"   Frequency: {info['cpu_freq_mhz']} MHz")
    print(f"   Usage: {info['cpu_usage_percent']}%")
    
    print(f"\n💾 MEMORY")
    print(f"   Total RAM: {info['ram_total_gb']} GB")
    print(f"   Available: {info['ram_available_gb']} GB")
    print(f"   Used: {info['ram_used_percent']}%")
    
    print(f"\n🎮 GPU")
    gpu = info['gpu_available']
    if gpu['has_gpu']:
        print(f"   Available: Yes")
        print(f"   Type: {gpu['gpu_type']}")
        print(f"   Name: {gpu['gpu_name']}")
        if gpu['gpu_count'] > 0:
            print(f"   Count: {gpu['gpu_count']}")
    else:
        print(f"   Available: No")
    
    print(f"\n🐍 PYTHON")
    print(f"   Version: {info['python_version']}")
    print("=" * 60)

if __name__ == "__main__":
    # Collect and display system info
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Optionally save to file
    import json
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    print("\n✅ System info saved to system_info.json")
