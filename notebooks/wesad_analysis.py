#!/usr/bin/env python3
"""
WESAD Results Analysis Script

Generates all figures and tables for the IEEE paper section on WESAD dataset.
Run this from the project root directory.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import analysis functions
from runs.baseline_analysis import analyze_baseline_experiment
from runs.dp_analysis import analyze_dp_experiment
from runs.fl_analysis import analyze_fl_experiment

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

sns.set_palette("husl")

def generate_latex_table(table_data, caption, label, filename):
    """Generate LaTeX table from data"""
    os.makedirs("paper/tables", exist_ok=True)
    filepath = f"paper/tables/{filename}"
    
    with open(filepath, 'w') as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(table_data)
        f.write("\\end{table}\n")
    
    print(f"📄 LaTeX table saved: {filepath}")

def save_publication_figure(fig, filename):
    """Save figure in both PNG and PDF formats"""
    os.makedirs("paper/figures", exist_ok=True)
    # PNG for quick viewing
    fig.savefig(f"paper/figures/{filename}.png", dpi=300, bbox_inches='tight')
    # PDF for publication
    fig.savefig(f"paper/figures/{filename}.pdf", dpi=300, bbox_inches='tight')
    print(f"📊 Figure saved: paper/figures/{filename}.(png|pdf)")

def main():
    print("🔍 WESAD Results Analysis")
    print(f"Working directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Baseline Analysis
    baseline_results_dir = "results/wesad/baseline"
    if os.path.exists(baseline_results_dir):
        print("\n🔍 Analyzing baseline experiments...")
        baseline_summary = analyze_baseline_experiment(baseline_results_dir, dataset="wesad")
        print("✅ Baseline analysis complete")
    else:
        print(f"❌ Baseline results directory not found: {baseline_results_dir}")
        baseline_summary = None
    
    # 2. DP Analysis
    dp_results_dir = "results/wesad/dp"
    if os.path.exists(dp_results_dir):
        print("\n🔒 Analyzing DP experiments...")
        dp_summary = analyze_dp_experiment(dp_results_dir, dataset="wesad")
        print("✅ DP analysis complete")
    else:
        print(f"❌ DP results directory not found: {dp_results_dir}")
        dp_summary = None
    
    # 3. FL Analysis
    fl_results_dir = "results/wesad/fl"
    if os.path.exists(fl_results_dir):
        print("\n🌐 Analyzing FL experiments...")
        fl_summary = analyze_fl_experiment(fl_results_dir, dataset="wesad")
        print("✅ FL analysis complete")
    else:
        print(f"❌ FL results directory not found: {fl_results_dir}")
        fl_summary = None
    
    # 4. Generate LaTeX Tables
    print("\n📄 Generating LaTeX tables...")
    
    # Baseline table
    if baseline_summary:
        stats = baseline_summary['statistics']
        table_data = f"""\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} \\\\
\\hline
Accuracy & {stats['accuracy']['mean']:.4f} & {stats['accuracy']['std']:.4f} \\\\
Precision & {stats['precision']['mean']:.4f} & {stats['precision']['std']:.4f} \\\\
Recall & {stats['recall']['mean']:.4f} & {stats['recall']['std']:.4f} \\\\
F1-Score & {stats['f1_score']['mean']:.4f} & {stats['f1_score']['std']:.4f} \\\\
\\hline
\\end{{tabular}}"""
        
        generate_latex_table(table_data, 
                            "Baseline performance statistics across 5 runs (WESAD)",
                            "tab:wesad_baseline_stats",
                            "wesad_baseline_stats.tex")
    
    # DP table
    if dp_summary:
        table_data = """\\begin{tabular}{|c|c|c|c|c|c|c|}
\\hline
\\textbf{Run} & \\textbf{$\\sigma$} & \\textbf{$\\epsilon$} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\hline
"""
        for run in dp_summary['runs']:
            table_data += f"{run['run']} & {run['noise_multiplier']:.1f} & {run['epsilon']:.2f} & {run['accuracy']:.4f} & {run['f1_score']:.4f} & {run['precision']:.4f} & {run['recall']:.4f} \\\\\n"
        table_data += "\\hline\n\\end{tabular}"
        
        generate_latex_table(table_data,
                            "Differential Privacy performance with varying noise multipliers (WESAD)",
                            "tab:wesad_dp_tradeoff",
                            "wesad_dp_tradeoff.tex")
    
    # FL table
    if fl_summary:
        table_data = """\\begin{tabular}{|c|c|c|c|c|}
\\hline
\\textbf{Clients} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\hline
"""
        for run in fl_summary['runs']:
            table_data += f"{run['num_clients']} & {run['accuracy']:.4f} & {run['f1_score']:.4f} & {run['precision']:.4f} & {run['recall']:.4f} \\\\\n"
        table_data += "\\hline\n\\end{tabular}"
        
        generate_latex_table(table_data,
                            "Federated Learning performance with different client numbers (WESAD)",
                            "tab:wesad_fl_performance",
                            "wesad_fl_performance.tex")
    
    # 5. Generate Figures
    print("\n📊 Generating publication-quality figures...")
    
    if baseline_summary:
        from runs.baseline_analysis import plot_baseline_runs
        fig = plot_baseline_runs(baseline_summary['runs'], "wesad")
        if fig:
            save_publication_figure(fig, "wesad_baseline_runs")
            plt.close(fig)
    
    if dp_summary:
        from runs.dp_analysis import plot_privacy_utility_tradeoff
        fig = plot_privacy_utility_tradeoff(dp_summary['runs'], "wesad")
        if fig:
            save_publication_figure(fig, "wesad_dp_tradeoff")
            plt.close(fig)
    
    if fl_summary:
        from runs.fl_analysis import plot_fl_performance
        fig = plot_fl_performance(fl_summary['runs'], "wesad")
        if fig:
            save_publication_figure(fig, "wesad_fl_performance")
            plt.close(fig)
    
    # 6. Generate Summary
    summary = {
        "dataset": "wesad",
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_summary,
        "differential_privacy": dp_summary,
        "federated_learning": fl_summary
    }
    
    with open("results/wesad/paper_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n📋 Comprehensive summary saved to: results/wesad/paper_summary.json")
    
    # Print key insights
    print("\n" + "="*80)
    print("WESAD RESULTS SUMMARY")
    print("="*80)
    
    if baseline_summary:
        stats = baseline_summary['statistics']
        print(f"Baseline (5 runs): Accuracy = {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
    
    if dp_summary:
        eps_range = dp_summary['statistics']['epsilon_range']
        acc_range = dp_summary['statistics']['accuracy_range']
        print(f"DP Range: ε ∈ [{eps_range['min']:.2f}, {eps_range['max']:.2f}], "
              f"Accuracy ∈ [{acc_range['min']:.4f}, {acc_range['max']:.4f}]")
    
    if fl_summary:
        client_range = fl_summary['statistics']['client_range']
        best_acc = fl_summary['statistics']['best_accuracy']
        print(f"FL Range: {client_range['min']}-{client_range['max']} clients, "
              f"Best Accuracy = {best_acc:.4f}")
    
    print("\n📄 Generated files:")
    print("- LaTeX tables: paper/tables/wesad_*.tex")
    print("- Figures: paper/figures/wesad_*.png|.pdf")
    print("- Summary: results/wesad/paper_summary.json")
    
    print("\n✅ WESAD analysis complete! Ready for IEEE paper.")

if __name__ == "__main__":
    main()
