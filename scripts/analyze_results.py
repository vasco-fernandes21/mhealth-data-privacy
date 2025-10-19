#!/usr/bin/env python3
"""
Analyze and report on all training results.

Usage:
    python scripts/analyze_results.py --results_dir ./results
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.privacy_utility_analysis import PrivacyUtilityAnalyzer
from src.utils.logging_utils import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('--results_dir', default='./results',
                       help='Results directory')
    parser.add_argument('--output_file', default='./results/analysis_report.txt',
                       help='Output report file')
    
    args = parser.parse_args()
    
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*70)
    logger.info("ANALYZING PRIVACY-UTILITY TRADEOFF RESULTS")
    logger.info("="*70)
    
    analyzer = PrivacyUtilityAnalyzer(args.results_dir)
    
    # Generate report
    analyzer.generate_report(args.output_file)
    
    logger.info(f"\n✅ Analysis complete!")
    logger.info(f"Report saved to: {args.output_file}")


if __name__ == "__main__":
    sys.exit(main())