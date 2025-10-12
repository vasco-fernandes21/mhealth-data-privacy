#!/usr/bin/env python3
"""
Script para executar análise completa localmente.
Uso: python scripts/run_analysis.py
"""

import os
import sys
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import load_all_results, calculate_comparison_metrics
from evaluation.visualization import create_comprehensive_analysis

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive analysis')
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--output_dir', default='./analysis', help='Analysis output directory')
    parser.add_argument('--dataset', choices=['sleep-edf', 'wesad', 'all'], default='all', help='Dataset to analyze')
    
    args = parser.parse_args()
    
    # Definir paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determinar datasets
    datasets = []
    if args.dataset == 'all':
        datasets = ['sleep-edf', 'wesad']
    else:
        datasets = [args.dataset]
    
    print("="*70)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    for dataset in datasets:
        print(f"\n📊 Analyzing {dataset} dataset...")
        
        dataset_results_dir = results_dir / dataset
        if not dataset_results_dir.exists():
            print(f"❌ Results not found: {dataset_results_dir}")
            continue
        
        try:
            # Carregar resultados
            results = load_all_results(str(dataset_results_dir))
            
            # Calcular métricas de comparação
            comparison_metrics = calculate_comparison_metrics(results)
            
            # Criar análise visual
            create_comprehensive_analysis(
                results=results,
                comparison_metrics=comparison_metrics,
                output_dir=str(output_dir / dataset),
                dataset_name=dataset
            )
            
            print(f"✅ Analysis completed for {dataset}!")
            
        except Exception as e:
            print(f"❌ Analysis failed for {dataset}: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
