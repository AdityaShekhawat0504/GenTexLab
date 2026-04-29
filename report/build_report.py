from __future__ import annotations

import argparse

from gentexlab.cli import latest_summary_path
from gentexlab.reporting import build_report_from_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GenTexLab experiment report")
    parser.add_argument("--summary", help="Path to summary.json. Defaults to latest run.")
    parser.add_argument("--output-dir", default="report", help="Directory for markdown and PDF outputs")
    args = parser.parse_args()

    summary_path = args.summary or str(latest_summary_path())
    outputs = build_report_from_summary(summary_path, output_dir=args.output_dir)
    print(f"Markdown report: {outputs['markdown']}")
    print(f"PDF report: {outputs['pdf']}")


if __name__ == "__main__":
    main()

