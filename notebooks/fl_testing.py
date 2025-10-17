#!/usr/bin/env python3
"""
Federated Learning Testing Script

Quick testing script for FL experiments before running full analysis.
Run this from the project root directory.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def test_fl_experiment(dataset="sleep-edf", num_clients=3, num_rounds=5, seed=42):
    """Test FL experiment with given parameters"""
    
    print(f"🧪 Testing FL for {dataset.upper()}")
    print(f"   Clients: {num_clients}, Rounds: {num_rounds}, Seed: {seed}")
    print("="*60)
    
    # Set environment variables
    os.environ['TRAIN_SEED'] = str(seed)
    os.environ['NUM_CLIENTS'] = str(num_clients)
    os.environ['NUM_ROUNDS'] = str(num_rounds)
    
    # Determine script path
    if dataset == "sleep-edf":
        script_path = "src/train/sleep-edf/federated-learning/train_fl.py"
        results_dir = f"results/sleep-edf/fl/fl_clients{num_clients}"
    elif dataset == "wesad":
        script_path = "src/train/wesad/federated-learning/train_fl.py"
        results_dir = f"results/wesad/fl/fl_clients{num_clients}"
    else:
        raise ValueError("Invalid dataset. Choose 'sleep-edf' or 'wesad'")
    
    # Run FL training
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=True, text=True, cwd=os.getcwd())
    training_time = time.time() - start_time
    
    print(f"⏱️  Training completed in {training_time:.1f}s")
    print(f"📊 Return code: {result.returncode}")
    
    if result.returncode == 0:
        # Load and display results
        results_file = os.path.join(results_dir, f"results_{dataset}_fl.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\n✅ SUCCESS - {dataset.upper()} FL Results:")
            print(f"   Accuracy:  {results['accuracy']:.4f}")
            print(f"   F1-Score:  {results['f1_score']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall:    {results['recall']:.4f}")
            print(f"   Training time: {results['training_time']:.1f}s")
            return True
        else:
            print(f"❌ Results file not found: {results_file}")
            return False
    else:
        print("❌ Training failed!")
        print("STDERR:", result.stderr)
        return False

if __name__ == "__main__":
    # Test configurations
    print("🚀 FL Testing Script")
    print("="*60)
    
    # Test Sleep-EDF
    print("\n1. Testing Sleep-EDF FL...")
    test_fl_experiment("sleep-edf", num_clients=3, num_rounds=5)
    
    # Test WESAD  
    print("\n2. Testing WESAD FL...")
    test_fl_experiment("wesad", num_clients=3, num_rounds=5)
    
    print("\n✅ FL testing complete!")
