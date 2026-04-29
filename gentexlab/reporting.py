from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from gentexlab.storage import ensure_dir, load_json


REPORT_TITLE = "Benchmarking Generative Models for Text-to-Texture Synthesis"


def build_report_from_summary(summary_path: str | Path, output_dir: str | Path = "report") -> dict[str, str]:
    summary = load_json(summary_path)
    run_dir = Path(summary["run_dir"])
    metrics_rows = load_json(run_dir / "metrics.json")
    output_path = ensure_dir(output_dir)
    markdown_path = output_path / "Benchmarking_Generative_Models_for_Text_to_Texture_Synthesis.md"
    pdf_path = output_path / "Benchmarking_Generative_Models_for_Text_to_Texture_Synthesis.pdf"

    markdown = _render_markdown(summary, metrics_rows)
    markdown_path.write_text(markdown, encoding="utf-8")
    _write_visual_pdf(pdf_path, summary, metrics_rows)

    return {"markdown": str(markdown_path), "pdf": str(pdf_path)}


def _render_markdown(summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> str:
    config = summary.get("config", {})
    recommendations = summary.get("recommendations", {})
    overall = recommendations.get("overall_ranking", [])
    category_rankings = recommendations.get("category_rankings", {})
    recommendation_text = recommendations.get("recommendations", {})
    metric_manifest = summary.get("metric_manifest", {})
    grouped_examples = _select_examples(metrics_rows)
    prompt_analysis = _prompt_sensitivity_analysis(metrics_rows)
    realism_analysis = _texture_realism_analysis(metrics_rows)
    tiling_analysis = _tiling_analysis(metrics_rows)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Methodology",
        "",
        "GenTexLab benchmarks text-to-texture generators using structured prompts, deterministic sampling, and a modular evaluation suite.",
        f"The analyzed run id is `{summary.get('run_id', 'unknown')}` and produced `{summary.get('num_records', 0)}` artifacts.",
        "",
        "### Experiment Configuration",
        "",
        f"- Experiment name: `{config.get('experiment_name', 'unknown')}`",
        f"- Models: `{', '.join(model['name'] for model in config.get('models', []))}`",
        f"- Prompt categories: `{', '.join(config.get('prompt_categories', []))}`",
        f"- Explicit prompts: `{len(config.get('prompts', []))}`",
        f"- Samples per prompt: `{config.get('num_samples', 0)}`",
        f"- Output resolution: `{config.get('width', 0)} x {config.get('height', 0)}`",
        "",
        "## Metrics Explanation",
        "",
        "- `clip_score`: semantic alignment between prompt text and generated image using CLIP when dependencies are available.",
        "- `seam_score`: left-right and top-bottom edge consistency for tileability.",
        "- `tiling_score`: continuity across repeated seams in a tiled layout.",
        "- `fft_score`: frequency-domain signal quality and repetitive artifact detection.",
        "- `stability_score`: consistency across multiple samples for the same prompt/model pair.",
        "",
        "## Metric Availability",
        "",
    ]

    for metric_name, manifest in metric_manifest.items():
        lines.append(
            f"- `{metric_name}`: available=`{manifest.get('available')}`, scope=`{manifest.get('scope')}`, note=`{manifest.get('last_message', '')}`"
        )

    lines.extend(["", "## Overall Comparison Results", ""])

    if overall:
        for rank, row in enumerate(overall, start=1):
            lines.append(
                f"{rank}. `{row['model_name']}` composite=`{_fmt(row.get('composite_score'))}` "
                f"clip=`{_fmt(row.get('clip_score'))}` seam=`{_fmt(row.get('seam_score'))}` "
                f"tiling=`{_fmt(row.get('tiling_score'))}` fft=`{_fmt(row.get('fft_score'))}` "
                f"stability=`{_fmt(row.get('stability_score'))}`"
            )
    else:
        lines.append("No ranking data available.")

    lines.extend(["", "## Category Recommendations", ""])
    if recommendation_text:
        for category, sentence in recommendation_text.items():
            lines.append(f"- {sentence}")
    else:
        lines.append("No category recommendations available.")

    lines.extend(["", "## Category Rankings", ""])
    if category_rankings:
        for category, ranking in category_rankings.items():
            lines.append(f"### {category.title()}")
            for rank, row in enumerate(ranking, start=1):
                lines.append(
                    f"{rank}. `{row['model_name']}` composite=`{_fmt(row.get('composite_score'))}` "
                    f"seam=`{_fmt(row.get('seam_score'))}` clip=`{_fmt(row.get('clip_score'))}`"
                )
            lines.append("")

    lines.extend(["## Metric Tables", ""])
    lines.extend(_markdown_metric_table(overall))
    lines.extend(["", "## Generated Image Examples", ""])
    for category, row in grouped_examples.items():
        lines.append(f"### {category.title()}")
        lines.append(f"Prompt: `{row['prompt']}`")
        lines.append(f"Texture:\n\n![{category} texture]({row['image_path']})")
        lines.append(f"Tiled 2x2:\n\n![{category} tiled]({row['tile_path']})")
        lines.append("")

    lines.extend(
        [
            "## Analysis",
            "",
            f"- Texture realism: {realism_analysis}",
            f"- Seamless tiling performance: {tiling_analysis}",
            f"- Prompt sensitivity: {prompt_analysis}",
            "",
        ]
    )

    lines.extend(
        [
            "## Insights",
            "",
            "The evaluation pipeline emphasizes tileability and reproducibility because those are often more operationally relevant in graphics production pipelines than visual appeal alone.",
            "When CLIP is available, semantic alignment can be weighted more strongly for concept-sensitive materials. When CLIP is unavailable, the recommendation engine still produces rankings from the remaining metrics.",
            "The procedural baseline is intended to validate infrastructure and should not be treated as a substitute for a learned diffusion model in final R&D comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "n/a"
    return str(value)


def _write_simple_pdf(path: Path, text: str) -> None:
    wrapped_lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(raw_line, width=92) or [""])

    lines_per_page = 46
    pages = [wrapped_lines[index : index + lines_per_page] for index in range(0, len(wrapped_lines), lines_per_page)] or [[]]

    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []

    for lines in pages:
        text_stream = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in lines:
            escaped = _pdf_escape(line)
            text_stream.append(f"({escaped}) Tj")
            text_stream.append("T*")
        text_stream.append("ET")
        stream_bytes = "\n".join(text_stream).encode("latin-1", errors="replace")
        content_id = add_object(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(0)

    pages_kids = []
    pages_id_placeholder = add_object(b"")
    for index, content_id in enumerate(content_ids):
        page_payload = (
            f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_ids[index] = add_object(page_payload)
        pages_kids.append(f"{page_ids[index]} 0 R")

    pages_payload = f"<< /Type /Pages /Kids [{' '.join(pages_kids)}] /Count {len(page_ids)} >>".encode("latin-1")
    objects[pages_id_placeholder - 1] = pages_payload
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id_placeholder} 0 R >>".encode("latin-1"))

    offsets = [0]
    body = bytearray(b"%PDF-1.4\n")
    for object_index, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{object_index} 0 obj\n".encode("latin-1"))
        body.extend(payload)
        body.extend(b"\nendobj\n")

    xref_position = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_position}\n%%EOF\n"
        ).encode("latin-1")
    )
    path.write_bytes(body)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _markdown_metric_table(overall: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Composite | CLIP | Seam | Tiling | FFT | Stability | Samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not overall:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in overall:
        lines.append(
            "| {model} | {comp} | {clip} | {seam} | {tiling} | {fft} | {stability} | {samples} |".format(
                model=row["model_name"],
                comp=_fmt(row.get("composite_score")),
                clip=_fmt(row.get("clip_score")),
                seam=_fmt(row.get("seam_score")),
                tiling=_fmt(row.get("tiling_score")),
                fft=_fmt(row.get("fft_score")),
                stability=_fmt(row.get("stability_score")),
                samples=row.get("sample_count", "n/a"),
            )
        )
    return lines


def _select_examples(metrics_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        category = row["prompt_category"]
        score = row.get("seam_score") or 0.0
        current = examples.get(category)
        if current is None or score > (current.get("seam_score") or 0.0):
            examples[category] = row
    return examples


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _texture_realism_analysis(metrics_rows: list[dict[str, Any]]) -> str:
    clip_values = [float(row["clip_score"]) for row in metrics_rows if isinstance(row.get("clip_score"), (int, float))]
    fft_values = [float(row["fft_score"]) for row in metrics_rows if isinstance(row.get("fft_score"), (int, float))]
    seam_values = [float(row["seam_score"]) for row in metrics_rows if isinstance(row.get("seam_score"), (int, float))]
    clip_mean = _safe_mean(clip_values)
    fft_mean = _safe_mean(fft_values)
    seam_mean = _safe_mean(seam_values)
    return (
        f"Mean CLIP alignment is {_fmt(clip_mean)}, mean FFT quality is {_fmt(fft_mean)}, "
        f"and mean seam continuity is {_fmt(seam_mean)}. Higher CLIP and seam values alongside "
        "moderate-to-high FFT values indicate more convincing, production-friendly texture structure."
    )


def _tiling_analysis(metrics_rows: list[dict[str, Any]]) -> str:
    tiling_values = [float(row["tiling_score"]) for row in metrics_rows if isinstance(row.get("tiling_score"), (int, float))]
    seam_values = [float(row["seam_score"]) for row in metrics_rows if isinstance(row.get("seam_score"), (int, float))]
    return (
        f"Across the pilot set, tiling consistency averages {_fmt(_safe_mean(tiling_values))} "
        f"and seam score averages {_fmt(_safe_mean(seam_values))}. The tiled previews help confirm "
        "whether numerical seam quality matches visual repetition behavior."
    )


def _prompt_sensitivity_analysis(metrics_rows: list[dict[str, Any]]) -> str:
    prompt_map: dict[str, list[float]] = {}
    for row in metrics_rows:
        if isinstance(row.get("clip_score"), (int, float)):
            prompt_map.setdefault(row["prompt"], []).append(float(row["clip_score"]))
    if not prompt_map:
        return "CLIP-based prompt sensitivity could not be computed because CLIP scores were unavailable for this run."
    spread = {
        prompt: (max(values) - min(values) if len(values) > 1 else 0.0)
        for prompt, values in prompt_map.items()
    }
    most_sensitive_prompt = max(spread, key=spread.get)
    return (
        f"The widest within-prompt CLIP spread is {_fmt(spread[most_sensitive_prompt])} for "
        f"`{most_sensitive_prompt}`, which makes it the most prompt-sensitive case in this pilot."
    )


def _write_visual_pdf(path: Path, summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    page_width = 1240
    page_height = 1754
    margin = 60
    bg = "white"
    text_color = "black"
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    pages: list[Image.Image] = []

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (page_width, page_height), bg)
        draw = ImageDraw.Draw(page)
        return page, draw, margin

    def add_wrapped(draw: ImageDraw.ImageDraw, y: int, text: str, font: Any, line_gap: int = 8) -> int:
        for paragraph in text.split("\n"):
            wrapped = textwrap.wrap(paragraph, width=120) or [""]
            for line in wrapped:
                draw.text((margin, y), line, fill=text_color, font=font)
                y += 18 + line_gap
            y += 6
        return y

    page, draw, y = new_page()
    y = add_wrapped(draw, y, REPORT_TITLE, title_font, line_gap=10)
    y = add_wrapped(draw, y, f"Run ID: {summary.get('run_id')}", body_font)
    y = add_wrapped(draw, y, f"Experiment: {summary.get('config', {}).get('experiment_name')}", body_font)
    y = add_wrapped(draw, y, "Overall ranking:", body_font)
    for row in summary.get("recommendations", {}).get("overall_ranking", []):
        line = (
            f"{row['model_name']} | composite={_fmt(row.get('composite_score'))} | "
            f"clip={_fmt(row.get('clip_score'))} | seam={_fmt(row.get('seam_score'))} | "
            f"tiling={_fmt(row.get('tiling_score'))} | fft={_fmt(row.get('fft_score'))} | "
            f"stability={_fmt(row.get('stability_score'))}"
        )
        y = add_wrapped(draw, y, line, body_font)
    pages.append(page)

    examples = list(_select_examples(metrics_rows).values())
    if examples:
        page, draw, y = new_page()
        y = add_wrapped(draw, y, "Generated image examples", title_font)
        x_positions = [margin, page_width // 2]
        row_height = 520
        for index, row in enumerate(examples[:4]):
            col = index % 2
            row_index = index // 2
            x = x_positions[col]
            current_y = y + row_index * row_height
            texture = Image.open(row["image_path"]).convert("RGB").resize((420, 420))
            tiled = Image.open(row["tile_path"]).convert("RGB").resize((220, 220))
            page.paste(texture, (x, current_y))
            page.paste(tiled, (x + 180, current_y + 190))
            draw.text((x, current_y + 430), f"{row['prompt_category']} | sample {row['sample_index']}", fill=text_color, font=body_font)
        pages.append(page)

    text_page, text_draw, y = new_page()
    y = add_wrapped(text_draw, y, "Analysis", title_font)
    y = add_wrapped(text_draw, y, _texture_realism_analysis(metrics_rows), body_font)
    y = add_wrapped(text_draw, y, _tiling_analysis(metrics_rows), body_font)
    y = add_wrapped(text_draw, y, _prompt_sensitivity_analysis(metrics_rows), body_font)
    pages.append(text_page)

    rgb_pages = [page.convert("RGB") for page in pages]
    _write_pdf_from_images(path, rgb_pages)


def _write_pdf_from_images(path: Path, images: list[Any]) -> None:
    import zlib

    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    page_ids: list[int] = []
    image_ids: list[int] = []
    content_ids: list[int] = []

    pages_id_placeholder = add_object(b"")

    for index, image in enumerate(images, start=1):
        rgb = image.convert("RGB")
        width, height = rgb.size
        raw = rgb.tobytes()
        compressed = zlib.compress(raw)
        image_payload = (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n".encode("latin-1")
            + compressed
            + b"\nendstream"
        )
        image_id = add_object(image_payload)
        image_ids.append(image_id)

        content_stream = (
            f"q\n{width} 0 0 {height} 0 0 cm\n/Im{index} Do\nQ\n".encode("latin-1")
        )
        content_payload = (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"endstream"
        )
        content_id = add_object(content_payload)
        content_ids.append(content_id)

        page_payload = (
            f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /XObject << /Im{index} {image_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_id = add_object(page_payload)
        page_ids.append(page_id)

    pages_payload = f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode(
        "latin-1"
    )
    objects[pages_id_placeholder - 1] = pages_payload
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id_placeholder} 0 R >>".encode("latin-1"))

    offsets = [0]
    body = bytearray(b"%PDF-1.4\n")
    for object_index, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{object_index} 0 obj\n".encode("latin-1"))
        body.extend(payload)
        body.extend(b"\nendobj\n")

    xref_position = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_position}\n%%EOF\n"
        ).encode("latin-1")
    )
    path.write_bytes(body)
