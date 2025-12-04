#!/usr/bin/env python3
"""
Baseline model comparisons: RF, XGBoost/GradientBoosting, SVM vs MLP.
Quick experiments to establish baseline performance.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_processed_data(dataset_name):
    """Load preprocessed data."""
    data_dir = Path(f'data/processed/{dataset_name}')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    return X_train, y_train, X_test, y_test

def run_random_forest(X_train, y_train, X_test, y_test, seed=42):
    """Train and evaluate Random Forest."""
    print(f"  Training Random Forest (seed={seed})...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }

def run_gradient_boosting(X_train, y_train, X_test, y_test, seed=42):
    """Train and evaluate Gradient Boosting (XGBoost alternative)."""
    print(f"  Training Gradient Boosting (seed={seed})...")
    start_time = time.time()
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed
    )
    
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }

def run_svm(X_train, y_train, X_test, y_test, seed=42):
    """Train and evaluate SVM with RBF kernel."""
    print(f"  Training SVM (seed={seed})...")
    start_time = time.time()
    
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=seed
    )
    
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }

def get_mlp_baseline_results(dataset_name):
    """Get existing MLP baseline results from aggregated metrics."""
    with open('results/aggregated_metrics.json', 'r') as f:
        data = json.load(f)
    
    baseline_key = f"baseline_{dataset_name}_Baseline"
    if baseline_key in data['detailed']:
        mlp_data = data['detailed'][baseline_key]
        return {
            'accuracy': mlp_data['accuracy']['mean'],
            'precision': mlp_data['precision']['mean'],
            'recall': mlp_data['recall']['mean'],
            'f1': mlp_data['f1_score']['mean'],
            'training_time': mlp_data['training']['training_time_seconds']['mean']
        }
    return None

if __name__ == '__main__':
    print("="*80)
    print("BASELINE MODEL COMPARISONS")
    print("="*80)
    
    results = {}
    
    for dataset_name in ['wesad', 'sleep-edf']:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load data
        print("\nLoading data...")
        X_train, y_train, X_test, y_test = load_processed_data(dataset_name)
        print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test: {X_test.shape[0]} samples")
        
        dataset_results = {}
        
        # Random Forest
        print("\n1. Random Forest")
        try:
            rf_results = run_random_forest(X_train, y_train, X_test, y_test)
            dataset_results['rf'] = rf_results
            print(f"  Accuracy: {rf_results['accuracy']*100:.2f}%")
            print(f"  Training time: {rf_results['training_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Gradient Boosting
        print("\n2. Gradient Boosting")
        try:
            gb_results = run_gradient_boosting(X_train, y_train, X_test, y_test)
            dataset_results['gb'] = gb_results
            print(f"  Accuracy: {gb_results['accuracy']*100:.2f}%")
            print(f"  Training time: {gb_results['training_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # SVM
        print("\n3. SVM (RBF kernel)")
        try:
            svm_results = run_svm(X_train, y_train, X_test, y_test)
            dataset_results['svm'] = svm_results
            print(f"  Accuracy: {svm_results['accuracy']*100:.2f}%")
            print(f"  Training time: {svm_results['training_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # MLP (from existing results)
        print("\n4. MLP (from existing experiments)")
        mlp_results = get_mlp_baseline_results(dataset_name)
        if mlp_results:
            dataset_results['mlp'] = mlp_results
            print(f"  Accuracy: {mlp_results['accuracy']*100:.2f}%")
            print(f"  Training time: {mlp_results['training_time']:.2f}s")
        else:
            print(f"  ✗ Could not load MLP results")
        
        results[dataset_name] = dataset_results
   