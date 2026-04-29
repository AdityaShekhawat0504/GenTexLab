from __future__ import annotations

import argparse
from pathlib import Path

from gentexlab.experiments import ExperimentRunner, load_experiment_config
from gentexlab.reporting import build_report_from_summary
from gentexlab.storage import load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenTexLab experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment config")
    run_parser.add_argument("config_path", help="Path to JSON or YAML experiment config")

    report_parser = subparsers.add_parser("report", help="Generate markdown and PDF report")
    report_parser.add_argument("--summary", help="Path to summary.json. Defaults to latest run.")
    report_parser.add_argument("--output-dir", default="report", help="Directory for report artifacts")

    summary_parser = subparsers.add_parser("summary", help="Print a short summary from summary.json")
    summary_parser.add_argument("--summary", help="Path to summary.json. Defaults to latest run.")

    return parser


def latest_summary_path(root: str | Path = "results/experiments") -> Path:
    base = Path(root)
    candidates = sorted(base.glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError("No experiment summaries found under results/experiments.")
    return candidates[-1]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = load_experiment_config(args.config_path)
        runner = ExperimentRunner()
        summary = runner.run(config)
        print(f"Run complete: {summary['run_dir']}")
        return

    if args.command == "report":
        summary_path = Path(args.summary) if args.summary else latest_summary_path()
        outputs = build_report_from_summary(summary_path, output_dir=args.output_dir)
        print(f"Report markdown: {outputs['markdown']}")
        print(f"Report pdf: {outputs['pdf']}")
        return

    if args.command == "summary":
        summary_path = Path(args.summary) if args.summary else latest_summary_path()
        summary = load_json(summary_path)
        print(f"Run id: {summary.get('run_id')}")
        print(f"Run dir: {summary.get('run_dir')}")
        overall = summary.get("recommendations", {}).get("overall_ranking", [])
        if overall:
            top = overall[0]
            print(f"Top model: {top['model_name']} (composite={top.get('composite_score')})")
        return


if __name__ == "__main__":
    main()

