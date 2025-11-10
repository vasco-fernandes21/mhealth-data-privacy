"""
Device utility functions for automatic hardware acceleration detection.
Supports hierarchical device selection: CUDA > CPU
"""

import torch


def get_optimal_device() -> torch.device:
    """
    Automatically detect and return the best available device for PyTorch training.

    Returns:
        torch.device: The optimal device (CUDA > CPU)
    """
    # Prioritize CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        # Default to CPU
        device = torch.device("cpu")
        print("Using CPU device")
        print("   Note: No GPU acceleration available")

    return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device information including availability and capabilities
    """
    import os

    # Configure optimal thread count for Apple Silicon
    if os.cpu_count() == 8 and torch.get_num_threads() < 8:
        # For M1/M2 chips with 8 cores, use all available threads
        torch.set_num_threads(8)
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = '8'

    info = {
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': torch.get_num_threads(),
        'physical_cpu_cores': os.cpu_count(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    if info['cuda_available']:
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'major': props.major,
                'minor': props.minor
            })

    return info


def print_device_info():
    """Print detailed information about available devices."""
    print("\n" + "=" * 50)
    print("DEVICE DETECTION REPORT")
    print("=" * 50)

    info = get_device_info()

    print(f"CUDA Available: {'Yes' if info['cuda_available'] else 'No'}")
    print(f"CPU Threads (Active): {info['cpu_count']}")
    print(f"Physical CPU Cores: {info['physical_cpu_cores']}")

    if info['cuda_available']:
        print(f"CUDA Devices: {info['device_count']}")
        for i, device in enumerate(info['cuda_devices']):
            print(f"   Device {i}: {device['name']}")
            print(f"   Memory: {device['total_memory_gb']:.1f} GB")
            print(f"   Compute Capability: {device['major']}.{device['minor']}")

    optimal_device = get_optimal_device()
    print(f"\nSelected Device: {optimal_device}")

    print("=" * 50)

    return optimal_device

