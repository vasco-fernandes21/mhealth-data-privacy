import subprocess
import json
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd

# ============================================================================
# 1️⃣ SETUP: Validate Project & Detect Device
# ============================================================================

print("="*70)
print("🚀 mHealth Privacy Experiments - Local Runner")
print("="*70)

PROJECT_DIR = Path.cwd()
RESULTS_DIR = PROJECT_DIR / 'experiments' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Validate project structure
required_paths = [
    'experiments/run_experiments.py',
    'data',
]

missing = [p for p in required_paths if not (PROJECT_DIR / p).exists()]
if missing:
    print("\n❌ ERROR: Missing project files:")
    for path in missing:
        print(f"   - {path}")
    sys.exit(1)

print(f"\n✅ Project validated at: {PROJECT_DIR}")

# Auto-detect device
print("\n📊 Device Detection:")
if torch.cuda.is_available():
    DEVICE = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   GPU: {gpu_name}")
    print(f"   VRAM: {vram:.1f} GB")
else:
    DEVICE = 'cpu'
    print(f"   Device: CPU")

# ============================================================================
# 2️⃣ CONFIG: Experiment Settings
# ============================================================================

print("\n" + "="*70)
print("⚙️  CONFIGURATION")
print("="*70)

# EDITA AQUI
SCENARIO = 'fl'           # Options: baseline, dp, fl, fl_dp, all
TAGS = ''                 # e.g., 'tier1' or '' (empty = no filter)
DATASETS = 'wesad'        # e.g., 'wesad', 'sleep-edf' or ''
N_EXPERIMENTS = None      # e.g., 3 or None (all)
DRY_RUN = False           # True = show commands only
AUTO_MODE = True          # True = skip confirmation

print(f"Scenario: {SCENARIO}")
print(f"Tags: {TAGS if TAGS else '(none)'}")
print(f"Datasets: {DATASETS if DATASETS else '(all)'}")
print(f"Limit experiments: {N_EXPERIMENTS if N_EXPERIMENTS else '(no limit)'}")
print(f"Dry run: {DRY_RUN}")
print(f"Auto mode: {AUTO_MODE}")

# ============================================================================
# 3️⃣ BUILD: Experiment Command
# ============================================================================

cmd = [
    sys.executable, '-u',
    'experiments/run_experiments.py',
    '--scenario', SCENARIO,
    '--device', DEVICE,
]

if TAGS:
    cmd += ['--tags', TAGS]
if DATASETS:
    cmd += ['--datasets', DATASETS]
if N_EXPERIMENTS:
    cmd += ['--n_experiments', str(N_EXPERIMENTS)]
if AUTO_MODE:
    cmd += ['--auto']

# ============================================================================
# 4️⃣ EXECUTE: Run Experiments
# ============================================================================

print("\n" + "="*70)
print("🚀 RUNNING EXPERIMENTS")
print("="*70)
print(f"\nCommand: {' '.join(cmd)}\n")

if not DRY_RUN:
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)

    # Wait for completion
    return_code = process.wait()

    print("\n" + "="*70)
    if return_code == 0:
        print("✅ Experiments completed successfully")
    else:
        print(f"❌ Experiments failed (exit code: {return_code})")
    print("="*70)
else:
    print("DRY RUN MODE - Command would execute above")

# ============================================================================
# 5️⃣ RESULTS: Load & Analyze
# ============================================================================

results_file = RESULTS_DIR / 'results_log.json'

if results_file.exists():
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)

    with open(results_file) as f:
        results = json.load(f)

    if 'timestamp' in results:
        print(f"\nTimestamp: {results['timestamp']}")

    if 'total' in results:
        print(f"Total experiments: {results['total']}")
        print(f"Successful: {results['successful']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Success rate: {results['successful']/results['total']*100:.1f}%")
        print(f"Total time: {results['total_time_hours']:.2f} hours")

        # Convert to DataFrame
        df = pd.DataFrame(results['results'])

        print(f"\nDetailed Results:")
        print(df[['name', 'method', 'dataset', 'seed', 'success',
                  'time_seconds']].to_string(index=False))

    print(f"\n{'='*70}")
else:
    print("\n⚠️  Results file not found")

# ============================================================================
# 6️⃣ UTILITIES: Helper Functions
# ============================================================================

def show_scenarios():
    """List available scenarios."""
    import yaml

    scenarios_dir = PROJECT_DIR / 'experiments' / 'scenarios'
    yaml_files = list(scenarios_dir.glob('*.yaml'))

    print(f'\n📋 Available Scenarios')
    for f in sorted(yaml_files):
        with open(f) as fp:
            data = yaml.safe_load(fp)
            n_exp = len(data.get('experiments', {}))
            print(f'  - {f.stem}: {n_exp} experiments')


def show_last_results():
    """Show last run results."""
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        df = pd.DataFrame(results['results'])
        print('\n📊 Last Run Results')
        print(df.to_string())
    else:
        print('No results available')


print("\n" + "="*70)
print("✅ Complete!")
print("="*70)
print("\nUsage:")
print("  show_scenarios()      # List all scenarios")
print("  show_last_results()   # Show last results")