# -*- coding: utf-8 -*-
"""Command-line entry point for the clinical analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_full_clinical_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full clinical analysis pipeline for MRI-derived clusters.",
    )
    parser.add_argument(
        "--cluster-file",
        type=Path,
        required=True,
        help="CSV with SubjectID and Cluster columns.",
    )
    parser.add_argument(
        "--clinical-dir",
        type=Path,
        required=True,
        help="Directory containing clinical_final CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tables/ and figures/ outputs.",
    )
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        help="Skip RF feature selection step.",
    )
    parser.add_argument(
        "--skip-logistic",
        action="store_true",
        help="Skip logistic models for binary outcomes.",
    )
    args = parser.parse_args()

    results = run_full_clinical_analysis(
        args.cluster_file,
        args.clinical_dir,
        args.output_dir,
        run_feature_selection=not args.skip_feature_selection,
        run_logistic=not args.skip_logistic,
    )
    print(f"Clinical analysis complete. Outputs written to: {args.output_dir}")
    print(f"Result keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
