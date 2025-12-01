"""Device utility functions for hardware acceleration detection."""

import torch
import os


def get_optimal_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    if os.cpu_count() == 8 and torch.get_num_threads() < 8:
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

