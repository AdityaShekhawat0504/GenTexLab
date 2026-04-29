from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def _load_runs(results_root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(results_root.glob("*/summary.json"), reverse=True):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_summary_path"] = str(summary_path)
        runs.append(summary)
    return runs


def _load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _to_table(data: list[dict]):
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return data
    return pd.DataFrame(data)


st.set_page_config(page_title="GenTexLab Dashboard", layout="wide")
st.title("GenTexLab")
st.caption("Generative Texture Evaluation & Integration Framework")

results_root = Path("results/experiments")
runs = _load_runs(results_root)

if not runs:
    st.warning("No experiment runs found in results/experiments.")
    st.stop()

selected_label = st.sidebar.selectbox(
    "Experiment Run",
    options=[f"{run['run_id']} | {run['config']['experiment_name']}" for run in runs],
)
selected_run = runs[[f"{run['run_id']} | {run['config']['experiment_name']}" for run in runs].index(selected_label)]
run_dir = Path(selected_run["run_dir"])
metrics = _load_metrics(run_dir)

models = sorted({row["model_name"] for row in metrics}) if metrics else []
categories = sorted({row["prompt_category"] for row in metrics}) if metrics else []

selected_models = st.sidebar.multiselect("Models", options=models, default=models)
selected_categories = st.sidebar.multiselect("Prompt Categories", options=categories, default=categories)

filtered = [
    row
    for row in metrics
    if row["model_name"] in selected_models and row["prompt_category"] in selected_categories
]

summary_col, rec_col = st.columns([1.2, 1.8])
with summary_col:
    st.subheader("Run Summary")
    st.json(
        {
            "run_id": selected_run["run_id"],
            "experiment_name": selected_run["config"]["experiment_name"],
            "num_records": selected_run["num_records"],
            "models": [model["name"] for model in selected_run["config"]["models"]],
            "metrics": selected_run["metrics_requested"],
        }
    )

with rec_col:
    st.subheader("Recommendations")
    for category, sentence in selected_run.get("recommendations", {}).get("recommendations", {}).items():
        if category in selected_categories:
            st.write(sentence)

st.subheader("Overall Ranking")
overall = selected_run.get("recommendations", {}).get("overall_ranking", [])
st.dataframe(_to_table(overall), use_container_width=True)

st.subheader("Category Ranking")
category_rankings = selected_run.get("recommendations", {}).get("category_rankings", {})
for category, ranking in category_rankings.items():
    if category in selected_categories:
        st.markdown(f"### {category.title()}")
        st.dataframe(_to_table(ranking), use_container_width=True)

st.subheader("Artifact Browser")
gallery_cols = st.columns(2)
for index, row in enumerate(filtered[:18]):
    with gallery_cols[index % 2]:
        st.image(row["image_path"], caption=f"{row['model_name']} | {row['prompt_category']} | sample {row['sample_index']}")
        st.image(row["tile_path"], caption="2x2 tiled preview")
        st.write(
            {
                "clip": row.get("clip_score"),
                "seam": row.get("seam_score"),
                "tiling": row.get("tiling_score"),
                "fft": row.get("fft_score"),
                "stability": row.get("stability_score"),
            }
        )

st.subheader("Metric Table")
st.dataframe(_to_table(filtered), use_container_width=True)
