#!/usr/bin/env python3
"""
Privacy-Utility Tradeoff Analysis.

Analyzes and plots the privacy-utility tradeoff.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class PrivacyUtilityAnalyzer:
    """Analyze privacy-utility tradeoff."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing results from all scenarios
        """
        self.results_dir = Path(results_dir)
    
    def load_results(self, scenario: str, dataset: str, param: Optional[str] = None) -> Dict:
        """
        Load results for a scenario.
        
        Args:
            scenario: 'baseline', 'dp', 'fl', 'fl_dp'
            dataset: 'sleep-edf' or 'wesad'
            param: Optional parameter (e.g., epsilon value)
        
        Returns:
            Results dictionary
        """
        if scenario == 'baseline':
            results_file = self.results_dir / scenario / dataset / 'results.json'
        elif scenario in ['dp', 'fl_dp']:
            results_file = self.results_dir / scenario / f'{param}' / dataset / 'results.json'
        else:
            results_file = self.results_dir / scenario / dataset / 'results.json'
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def create_comparison_table(self, dataset: str) -> Dict[str, List]:
        """
        Create comparison table across all scenarios.
        
        Args:
            dataset: Dataset name
        
        Returns:
            Comparison table
        """
        comparison = {
            'scenario': [],
            'privacy_param': [],
            'accuracy': [],
            'f1_score': [],
            'training_time': [],
            'privacy_budget': []
        }
        
        # Baseline
        baseline_results = self.load_results('baseline', dataset)
        if baseline_results:
            comparison['scenario'].append('Baseline')
            comparison['privacy_param'].append('None')
            comparison['accuracy'].append(baseline_results.get('accuracy', np.nan))
            comparison['f1_score'].append(baseline_results.get('f1_score', np.nan))
            comparison['training_time'].append(baseline_results.get('training_time_seconds', np.nan))
            comparison['privacy_budget'].append(np.inf)
        
        # DP with different epsilons
        for epsilon in [0.5, 1.0, 5.0, 10.0]:
            dp_results = self.load_results('dp', dataset, f'epsilon_{epsilon}')
            if dp_results:
                comparison['scenario'].append('DP')
                comparison['privacy_param'].append(f'ε={epsilon}')
                comparison['accuracy'].append(dp_results.get('accuracy', np.nan))
                comparison['f1_score'].append(dp_results.get('f1_score', np.nan))
                comparison['training_time'].append(dp_results.get('training_time_seconds', np.nan))
                comparison['privacy_budget'].append(dp_results.get('final_epsilon', np.nan))
        
        # FL
        fl_results = self.load_results('fl', dataset)
        if fl_results:
            comparison['scenario'].append('FL')
            comparison['privacy_param'].append('None')
            comparison['accuracy'].append(fl_results.get('accuracy', np.nan))
            comparison['f1_score'].append(fl_results.get('f1_score', np.nan))
            comparison['training_time'].append(fl_results.get('training_time_seconds', np.nan))
            comparison['privacy_budget'].append(np.inf)
        
        # FL+DP
        for epsilon in [1.0, 5.0]:
            fl_dp_results = self.load_results('fl_dp', dataset, f'epsilon_{epsilon}')
            if fl_dp_results:
                comparison['scenario'].append('FL+DP')
                comparison['privacy_param'].append(f'ε={epsilon}')
                comparison['accuracy'].append(fl_dp_results.get('accuracy', np.nan))
                comparison['f1_score'].append(fl_dp_results.get('f1_score', np.nan))
                comparison['training_time'].append(fl_dp_results.get('training_time_seconds', np.nan))
                comparison['privacy_budget'].append(fl_dp_results.get('final_epsilon', np.nan))
        
        return comparison
    
    def generate_report(self, output_file: str) -> None:
        """
        Generate analysis report.
        
        Args:
            output_file: Path to save report
        """
        report = []
        report.append("="*80)
        report.append("PRIVACY-UTILITY TRADEOFF ANALYSIS")
        report.append("="*80)
        
        for dataset in ['sleep-edf', 'wesad']:
            report.append(f"\nDataset: {dataset}")
            report.append("-"*80)
            
            comparison = self.create_comparison_table(dataset)
            
            # Create formatted table
            report.append(f"\n{'Scenario':<15} {'Parameter':<15} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<12}")
            report.append("-"*70)
            
            for i in range(len(comparison['scenario'])):
                scenario = comparison['scenario'][i]
                param = comparison['privacy_param'][i]
                acc = comparison['accuracy'][i]
                f1 = comparison['f1_score'][i]
                time_s = comparison['training_time'][i]
                
                acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
                time_str = f"{time_s:.1f}" if not np.isnan(time_s) else "N/A"
                
                report.append(f"{scenario:<15} {param:<15} {acc_str:<12} {f1_str:<12} {time_str:<12}")
        
        # Summary
        report.append("\n" + "="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        report.append("• Baseline provides upper bound on accuracy (no privacy)")
        report.append("• DP trading off accuracy for privacy guarantees")
        report.append("• FL enabling distributed training without centralized data")
        report.append("• FL+DP combining both for privacy-preserving federated learning")
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    print("Privacy-Utility Analysis module loaded")