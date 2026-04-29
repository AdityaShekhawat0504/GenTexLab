from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
import time
from typing import Any

from gentexlab.evaluation import build_metrics
from gentexlab.experiments.recommendation import build_recommendations
from gentexlab.experiments.schema import ExperimentConfig
from gentexlab.experiments.types import GeneratedSampleRecord
from gentexlab.models import create_generator
from gentexlab.prompts import build_prompt_specs
from gentexlab.storage import ensure_dir, save_csv, save_json, slugify, timestamp_run_id


class ExperimentRunner:
    def run(self, config: ExperimentConfig) -> dict[str, Any]:
        run_id = timestamp_run_id(config.experiment_name)
        run_dir = ensure_dir(Path(config.output_root) / run_id)
        images_dir = ensure_dir(run_dir / "images")
        tiles_dir = ensure_dir(run_dir / "tiles")
        log_dir = ensure_dir(run_dir / "logs")
        logger = self._create_logger(log_dir / "experiment.log")
        logger.info("Starting experiment '%s'", config.experiment_name)

        prompt_specs = build_prompt_specs(prompts=config.prompts, categories=config.prompt_categories)
        logger.info("Resolved %s structured prompts.", len(prompt_specs))

        metrics = build_metrics(config.metrics)
        metric_manifest: dict[str, dict[str, Any]] = {}
        records: list[GeneratedSampleRecord] = []

        for model_index, model_config in enumerate(config.models):
            generator = create_generator(model_config)
            logger.info("Running model '%s' (%s)", model_config.name, model_config.model_id or model_config.name)
            for prompt_index, prompt_spec in enumerate(prompt_specs):
                for sample_index in range(config.num_samples):
                    seed = config.seed + model_index * 10_000 + prompt_index * 100 + sample_index
                    logger.info(
                        "Generating model=%s prompt_index=%s sample_index=%s seed=%s prompt=%s",
                        model_config.name,
                        prompt_index,
                        sample_index,
                        seed,
                        prompt_spec.structured_prompt,
                    )
                    started = time.perf_counter()
                    output = generator.generate(
                        prompt=prompt_spec.structured_prompt,
                        category=prompt_spec.category,
                        width=config.width,
                        height=config.height,
                        seed=seed,
                        negative_prompt=config.negative_prompt,
                    )
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "Generated artifact for model=%s prompt_index=%s sample_index=%s in %.2fs",
                        model_config.name,
                        prompt_index,
                        sample_index,
                        elapsed,
                    )
                    artifact_id = (
                        f"{slugify(model_config.name)}-{slugify(prompt_spec.category)}-"
                        f"{prompt_index:03d}-{sample_index:02d}"
                    )
                    image_path = images_dir / f"{artifact_id}.png"
                    tile_path = tiles_dir / f"{artifact_id}-tile2x2.png"
                    output.image.save_png(image_path)
                    output.image.tile_2x2().save_png(tile_path)
                    record = GeneratedSampleRecord(
                        artifact_id=artifact_id,
                        experiment_name=config.experiment_name,
                        model_name=model_config.name,
                        model_id=generator.model_id,
                        prompt=prompt_spec.structured_prompt,
                        raw_prompt=prompt_spec.raw_prompt,
                        prompt_category=prompt_spec.category,
                        sample_index=sample_index,
                        seed=seed,
                        image_path=str(image_path),
                        tile_path=str(tile_path),
                        generation_seconds=elapsed,
                        image=output.image,
                        generation_metadata=output.metadata,
                    )
                    records.append(record)

        rows: list[dict[str, Any]] = []
        for record in records:
            row = record.base_row()
            for metric in metrics:
                result = metric.evaluate(record, records=records)
                row[metric.name] = result.score
                row[f"{metric.name}_available"] = result.available
                if result.message:
                    row[f"{metric.name}_message"] = result.message
                record.metric_details[metric.name] = result.details
                metric_manifest.setdefault(metric.name, {})
                metric_manifest[metric.name].update(
                    {
                        "scope": metric.scope,
                        "available": result.available,
                        "last_message": result.message,
                    }
                )
            row["generation_metadata"] = record.generation_metadata
            row["metric_details"] = record.metric_details
            rows.append(row)

        recommendations = build_recommendations(rows, config.recommendation_weights)
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "config": asdict(config),
            "num_records": len(records),
            "metrics_requested": config.metrics,
            "metric_manifest": metric_manifest,
            "recommendations": recommendations,
        }

        save_csv(run_dir / "metrics.csv", self._flatten_rows(rows))
        save_json(run_dir / "metrics.json", rows)
        save_json(run_dir / "summary.json", summary)
        save_json(
            run_dir / "artifacts.json",
            [
                {
                    **record.base_row(),
                    "generation_metadata": record.generation_metadata,
                    "metric_details": record.metric_details,
                }
                for record in records
            ],
        )
        logger.info("Completed experiment '%s' with %s artifacts.", config.experiment_name, len(records))
        return summary

    def _create_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger(f"gentexlab.{log_path.stem}.{id(log_path)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def _flatten_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for row in rows:
            copy = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    copy[key] = str(value)
                else:
                    copy[key] = value
            flattened.append(copy)
        return flattened
