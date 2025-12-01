#!/usr/bin/env python3
"""
Comprehensive Results Aggregation Script
Extracts and aggregates all metrics from experiment results for paper analysis.
"""

import os
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

EXPERIMENTS_DIR = "results/experiments"
OUTPUT_FILE = "results/aggregated_metrics.json"

def load_json(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def safe_mean(values):
    """Calculate mean, handling None values and type conversion."""
    valid = []
    for v in values:
        if v is not None:
            try:
                valid.append(float(v))
            except (ValueError, TypeError):
                continue
    return statistics.mean(valid) if valid else None

def safe_stdev(values):
    """Calculate standard deviation, handling None values and type conversion."""
    valid = []
    for v in values:
        if v is not None:
            try:
                valid.append(float(v))
            except (ValueError, TypeError):
                continue
    return statistics.stdev(valid) if len(valid) > 1 else 0.0

def extract_config_from_filename(filename, method, data):
    """Extract configuration parameters from filename or data."""
    config = {}
    
    if method == 'dp':
        # Try hyperparameters first
        sigma = data.get('hyperparameters', {}).get('noise_multiplier')
        if sigma is None:
            # Try parsing filename
            if "noise" in filename:
                try:
                    parts = filename.split("noise")[1].split("_")[0]
                    sigma = float(parts.replace("p", "."))
                except:
                    pass
        config['noise_multiplier'] = sigma
        config['max_grad_norm'] = data.get('hyperparameters', {}).get('max_grad_norm', 1.0)
        
    elif method == 'fl':
        n_clients = data.get('hyperparameters', {}).get('n_clients')
        if n_clients is None:
            # Try parsing filename
            if "_c" in filename or "clients" in filename:
                try:
                    parts = filename.split("_c")[1] if "_c" in filename else filename.split("clients")[1]
                    n_clients = int(parts.split("_")[0])
                except:
                    pass
        config['n_clients'] = n_clients
        config['local_epochs'] = data.get('hyperparameters', {}).get('local_epochs', 1)
        
    elif method in ['dp_fl', 'fl_dp']:
        n_clients = data.get('hyperparameters', {}).get('n_clients')
        sigma = data.get('hyperparameters', {}).get('noise_multiplier')
        config['n_clients'] = n_clients
        config['noise_multiplier'] = sigma
        config['max_grad_norm'] = data.get('hyperparameters', {}).get('max_grad_norm', 1.0)
        config['local_epochs'] = data.get('hyperparameters', {}).get('local_epochs', 1)
    
    return config

def aggregate_experiments():
    """Aggregate all experiment results with comprehensive metrics."""
    results = defaultdict(list)
    
    # Walk through the directory
    for root, dirs, files in os.walk(EXPERIMENTS_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue
                
            filepath = os.path.join(root, file)
            data = load_json(filepath)
            
            if not data:
                continue
                
            info = data.get('experiment_info', {})
            method = info.get('method', 'unknown')
            dataset = info.get('dataset', 'unknown')
            seed = info.get('seed', 'unknown')
            
            # Skip if missing info
            if method == 'unknown' or dataset == 'unknown':
                continue
            
            # Extract configuration
            config = extract_config_from_filename(file, method, data)
            
            # Create config key for grouping
            if method == 'baseline':
                config_key = "Baseline"
            elif method == 'dp':
                sigma = config.get('noise_multiplier', '?')
                config_key = f"DP_σ={sigma}"
            elif method == 'fl':
                n = config.get('n_clients', '?')
                config_key = f"FL_N={n}"
            elif method in ['dp_fl', 'fl_dp']:
                n = config.get('n_clients', '?')
                sigma = config.get('noise_multiplier', '?')
                config_key = f"FL+DP_N={n}_σ={sigma}"
            else:
                config_key = method
            
            # Extract test metrics
            test_metrics = data.get('test_metrics', {})
            training_metrics = data.get('training_metrics', {})
            privacy_metrics = data.get('privacy_metrics', {})
            hyperparams = data.get('hyperparameters', {})
            timing = data.get('timing', {})
            
            # Extract all relevant metrics
            run_data = {
                'seed': seed,
                'filename': file,
                
                # Test metrics
                'accuracy': test_metrics.get('accuracy'),
                'precision': test_metrics.get('precision'),
                'recall': test_metrics.get('recall'),
                'f1_score': test_metrics.get('f1_score'),
                
                # Per-class metrics (for multi-class)
                'precision_per_class': test_metrics.get('precision_per_class', []),
                'recall_per_class': test_metrics.get('recall_per_class', []),
                'f1_per_class': test_metrics.get('f1_per_class', []),
                
                # Training metrics
                'total_epochs': training_metrics.get('total_epochs'),
                'best_epoch': training_metrics.get('best_epoch'),
                'best_val_acc': training_metrics.get('best_val_acc'),
                'training_time_seconds': training_metrics.get('training_time_seconds'),
                'early_stopped': training_metrics.get('convergence', {}).get('early_stopped', False),
                
                # Privacy metrics
                'final_epsilon': privacy_metrics.get('final_epsilon'),
                'delta': privacy_metrics.get('delta'),
                'noise_multiplier': privacy_metrics.get('noise_multiplier') or config.get('noise_multiplier'),
                'max_grad_norm': privacy_metrics.get('max_grad_norm') or config.get('max_grad_norm'),
                
                # FL-specific
                'n_clients': privacy_metrics.get('n_clients') or config.get('n_clients'),
                'total_rounds': privacy_metrics.get('total_rounds') or training_metrics.get('total_epochs'),
                'local_epochs': privacy_metrics.get('local_epochs') or config.get('local_epochs'),
                
                # Timing
                'total_time_seconds': timing.get('total_time_seconds'),
                
                # Configuration
                'config': config
            }
            
            results[(method, dataset, config_key)].append(run_data)
    
    # Calculate aggregated statistics
    aggregated = {}
    
    for (method, dataset, config_key), runs in results.items():
        if not runs:
            continue
        
        # Extract all values
        accuracies = [r['accuracy'] for r in runs if r['accuracy'] is not None]
        precisions = [r['precision'] for r in runs if r['precision'] is not None]
        recalls = [r['recall'] for r in runs if r['recall'] is not None]
        f1_scores = [r['f1_score'] for r in runs if r['f1_score'] is not None]
        
        # Per-class metrics (aggregate across runs)
        all_precision_per_class = [r['precision_per_class'] for r in runs if r['precision_per_class']]
        all_recall_per_class = [r['recall_per_class'] for r in runs if r['recall_per_class']]
        all_f1_per_class = [r['f1_per_class'] for r in runs if r['f1_per_class']]
        
        # Privacy metrics
        epsilons = [r['final_epsilon'] for r in runs if r['final_epsilon'] is not None]
        deltas = [r['delta'] for r in runs if r['delta'] is not None]
        noise_multipliers = [r['noise_multiplier'] for r in runs if r['noise_multiplier'] is not None]
        max_grad_norms = [r['max_grad_norm'] for r in runs if r['max_grad_norm'] is not None]
        
        # Training metrics
        total_epochs = [r['total_epochs'] for r in runs if r['total_epochs'] is not None]
        best_val_accs = [r['best_val_acc'] for r in runs if r['best_val_acc'] is not None]
        training_times = [r['training_time_seconds'] for r in runs if r['training_time_seconds'] is not None]
        
        # FL metrics
        n_clients_list = [r['n_clients'] for r in runs if r['n_clients'] is not None]
        total_rounds_list = [r['total_rounds'] for r in runs if r['total_rounds'] is not None]
        
        # Aggregate per-class metrics
        precision_per_class_agg = None
        recall_per_class_agg = None
        f1_per_class_agg = None
        
        if all_precision_per_class and len(all_precision_per_class[0]) > 0:
            n_classes = len(all_precision_per_class[0])
            precision_per_class_agg = []
            for i in range(n_classes):
                values = []
                for run in all_precision_per_class:
                    if i < len(run) and run[i] is not None:
                        try:
                            values.append(float(run[i]))
                        except (ValueError, TypeError):
                            pass
                precision_per_class_agg.append(safe_mean(values))
            
            recall_per_class_agg = None
            if all_recall_per_class:
                recall_per_class_agg = []
                for i in range(n_classes):
                    values = []
                    for run in all_recall_per_class:
                        if i < len(run) and run[i] is not None:
                            try:
                                values.append(float(run[i]))
                            except (ValueError, TypeError):
                                pass
                    recall_per_class_agg.append(safe_mean(values))
            
            f1_per_class_agg = None
            if all_f1_per_class:
                f1_per_class_agg = []
                for i in range(n_classes):
                    values = []
                    for run in all_f1_per_class:
                        if i < len(run) and run[i] is not None:
                            try:
                                values.append(float(run[i]))
                            except (ValueError, TypeError):
                                pass
                    f1_per_class_agg.append(safe_mean(values))
        
        # Build aggregated result
        aggregated[f"{method}_{dataset}_{config_key}"] = {
            "method": method,
            "dataset": dataset,
            "config_key": config_key,
            "n_runs": len(runs),
            "seeds": [r['seed'] for r in runs],
            
            # Overall metrics
            "accuracy": {
                "mean": safe_mean(accuracies),
                "std": safe_stdev(accuracies),
                "min": min(accuracies) if accuracies else None,
                "max": max(accuracies) if accuracies else None,
                "values": accuracies
            },
            "precision": {
                "mean": safe_mean(precisions),
                "std": safe_stdev(precisions),
                "values": precisions
            },
            "recall": {
                "mean": safe_mean(recalls),
                "std": safe_stdev(recalls),
                "values": recalls
            },
            "f1_score": {
                "mean": safe_mean(f1_scores),
                "std": safe_stdev(f1_scores),
                "values": f1_scores
            },
            
            # Per-class metrics
            "precision_per_class": precision_per_class_agg,
            "recall_per_class": recall_per_class_agg,
            "f1_per_class": f1_per_class_agg,
            
            # Privacy metrics
            "privacy": {
                "epsilon": {
                    "mean": safe_mean(epsilons),
                    "std": safe_stdev(epsilons),
                    "min": min(epsilons) if epsilons else None,
                    "max": max(epsilons) if epsilons else None,
                    "values": epsilons
                },
                "delta": safe_mean(deltas) if deltas else None,
                "noise_multiplier": safe_mean(noise_multipliers) if noise_multipliers else None,
                "max_grad_norm": safe_mean(max_grad_norms) if max_grad_norms else None
            },
            
            # Training metrics
            "training": {
                "total_epochs": {
                    "mean": safe_mean(total_epochs),
                    "values": total_epochs
                },
                "best_val_acc": {
                    "mean": safe_mean(best_val_accs),
                    "std": safe_stdev(best_val_accs),
                    "values": best_val_accs
                },
                "training_time_seconds": {
                    "mean": safe_mean(training_times),
                    "total": sum(training_times) if training_times else None,
                    "values": training_times
                }
            },
            
            # FL-specific
            "federated": {
                "n_clients": safe_mean(n_clients_list) if n_clients_list else None,
                "total_rounds": {
                    "mean": safe_mean(total_rounds_list),
                    "values": total_rounds_list
                }
            } if method in ['fl', 'dp_fl', 'fl_dp'] else None
        }
    
    return aggregated

def generate_summary_table(aggregated):
    """Generate a summary table for the paper."""
    summary = {
        "baseline": {},
        "dp": {},
        "fl": {},
        "dp_fl": {}
    }
    
    for key, data in aggregated.items():
        method = data['method']
        dataset = data['dataset']
        config = data['config_key']
        
        if method == 'baseline':
            summary['baseline'][dataset] = {
                "accuracy": data['accuracy']['mean'],
                "recall": data['recall']['mean'],
                "f1": data['f1_score']['mean'],
                "training_time_seconds": data['training']['training_time_seconds']['mean'],
                "n_runs": data['n_runs']
            }
        elif method == 'dp':
            if dataset not in summary['dp']:
                summary['dp'][dataset] = {}
            summary['dp'][dataset][config] = {
                "accuracy": data['accuracy']['mean'],
                "recall": data['recall']['mean'],
                "epsilon": data['privacy']['epsilon']['mean'],
                "f1": data['f1_score']['mean'],
                "training_time_seconds": data['training']['training_time_seconds']['mean'],
                "n_runs": data['n_runs']
            }
        elif method == 'fl':
            if dataset not in summary['fl']:
                summary['fl'][dataset] = {}
            summary['fl'][dataset][config] = {
                "accuracy": data['accuracy']['mean'],
                "recall": data['recall']['mean'],
                "f1": data['f1_score']['mean'],
                "training_time_seconds": data['training']['training_time_seconds']['mean'],
                "n_runs": data['n_runs']
            }
        elif method in ['dp_fl', 'fl_dp']:
            if dataset not in summary['dp_fl']:
                summary['dp_fl'][dataset] = {}
            summary['dp_fl'][dataset][config] = {
                "accuracy": data['accuracy']['mean'],
                "recall": data['recall']['mean'],
                "epsilon": data['privacy']['epsilon']['mean'],
                "f1": data['f1_score']['mean'],
                "training_time_seconds": data['training']['training_time_seconds']['mean'],
                "n_runs": data['n_runs']
            }
    
    return summary

if __name__ == "__main__":
    print("Aggregating experiment results...")
    agg_results = aggregate_experiments()
    
    # Generate summary
    summary = generate_summary_table(agg_results)
    
    output = {
        "summary": summary,
        "detailed": agg_results
    }
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAggregated {len(agg_results)} experiment configurations")
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for method in ['baseline', 'dp', 'fl', 'dp_fl']:
        if summary[method]:
            print(f"\n{method.upper()}:")
            for dataset, configs in summary[method].items():
                print(f"  {dataset}:")
                if isinstance(configs, dict) and 'accuracy' in configs:
                    # Baseline
                    time_str = f", Time={configs.get('training_time_seconds', 0):.1f}s" if configs.get('training_time_seconds') else ""
                    print(f"    Accuracy: {configs['accuracy']:.4f}, Recall: {configs.get('recall', 0):.4f}, F1: {configs['f1']:.4f}{time_str}")
                else:
                    # Other methods
                    for config, metrics in configs.items():
                        eps_str = f", ε={metrics['epsilon']:.2f}" if metrics.get('epsilon') else ""
                        time_str = f", Time={metrics.get('training_time_seconds', 0):.1f}s" if metrics.get('training_time_seconds') else ""
                        print(f"    {config}: Acc={metrics['accuracy']:.4f}, Recall={metrics.get('recall', 0):.4f}, F1={metrics['f1']:.4f}{eps_str}{time_str}")
