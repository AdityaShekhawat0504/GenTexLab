# GenTexLab

GenTexLab is a research-oriented framework for benchmarking text-to-texture generation systems. It separates prompt construction, model execution, evaluation, result storage, recommendation, dashboarding, and report generation so teams can compare generative models with reproducible experiment definitions.

## What It Does

- Runs benchmark experiments from JSON or YAML configs
- Supports multiple generator backends through a shared interface
- Ships with a deterministic procedural baseline for local validation
- Integrates Diffusers-based Stable Diffusion and SDXL pipelines when dependencies are installed
- Evaluates outputs with semantic, seamlessness, tiling, frequency, and stability metrics
- Saves generated assets, structured metrics, summaries, and recommendations
- Exposes a Streamlit dashboard for browsing results and model rankings
- Builds a PDF report titled `Benchmarking Generative Models for Text-to-Texture Synthesis`

## Repository Layout

```text
gentexlab/
├── models/              # model adapters and inference backends
├── prompts/             # structured prompt templates and category logic
├── evaluation/          # evaluation metrics
├── experiments/         # config schema, runner, recommendations
├── dashboard/           # Streamlit app
├── results/             # generated artifacts and experiment outputs
├── configs/             # sample experiment configurations
└── report/              # generated markdown/PDF reports
```

## Quick Start

1. Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a benchmark:

```bash
python3 -m gentexlab.cli run configs/sample_experiment.json
```

3. Generate a report for the latest run:

```bash
python3 -m gentexlab.cli report
```

4. Launch the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

5. Run the lightweight validation suite:

```bash
python3 -m unittest discover -s tests
```

## Experiment Config

GenTexLab accepts JSON directly and YAML when `PyYAML` is installed.

```json
{
  "experiment_name": "texture-benchmark",
  "models": [
    "procedural-baseline",
    "stable-diffusion",
    "sdxl"
  ],
  "prompt_categories": ["wood", "metal", "fabric", "stone", "plastic"],
  "num_samples": 3,
  "metrics": ["clip_score", "seam_score", "tiling_score", "fft_score", "stability_score"],
  "seed": 42,
  "width": 512,
  "height": 512
}
```

## Notes On Dependencies

- `procedural-baseline` runs without PyTorch or Diffusers and is useful for validating the experiment stack end to end.
- `stable-diffusion`, `sdxl`, and `sdxl-turbo` require `torch`, `diffusers`, and compatible runtime acceleration.
- `clip_score` uses CLIP through `transformers` and `torch` when available. If CLIP dependencies are missing, the metric is marked unavailable instead of failing the run.
- The frequency and tiling metrics include pure-Python fallbacks so the framework can still run in constrained environments.

## Reproducibility

- Experiment identity is derived from config plus a timestamped run directory
- Seeds are deterministic across model, prompt, and sample index
- Metrics, recommendations, manifests, and summary artifacts are stored with each experiment
- Report generation reads directly from structured experiment outputs
