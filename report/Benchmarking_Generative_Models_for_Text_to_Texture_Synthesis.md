# Benchmarking Generative Models for Text-to-Texture Synthesis

## Methodology

GenTexLab benchmarks text-to-texture generators using structured prompts, deterministic sampling, and a modular evaluation suite.
The analyzed run id is `20260429-130504-real-diffusion-run` and produced `9` artifacts.

### Experiment Configuration

- Experiment name: `real-diffusion-run`
- Models: `stable-diffusion`
- Prompt categories: ``
- Explicit prompts: `3`
- Samples per prompt: `3`
- Output resolution: `512 x 512`

## Metrics Explanation

- `clip_score`: semantic alignment between prompt text and generated image using CLIP when dependencies are available.
- `seam_score`: left-right and top-bottom edge consistency for tileability.
- `tiling_score`: continuity across repeated seams in a tiled layout.
- `fft_score`: frequency-domain signal quality and repetitive artifact detection.
- `stability_score`: consistency across multiple samples for the same prompt/model pair.

## Metric Availability

- `clip_score`: available=`True`, scope=`image`, note=``
- `seam_score`: available=`True`, scope=`image`, note=``
- `tiling_score`: available=`True`, scope=`image`, note=``
- `fft_score`: available=`True`, scope=`image`, note=``
- `stability_score`: available=`True`, scope=`group`, note=``

## Overall Comparison Results

1. `stable-diffusion` composite=`0.622` clip=`0.298` seam=`0.851` tiling=`0.804` fft=`0.654` stability=`0.784`

## Category Recommendations

- For wood textures: stable-diffusion performs best with composite score 0.663 (clip_score=0.326, seam_score=0.919, tiling_score=0.885, fft_score=0.548, stability_score=0.870).
- For metal textures: stable-diffusion performs best with composite score 0.587 (clip_score=0.267, seam_score=0.804, tiling_score=0.748, fft_score=0.684, stability_score=0.745).
- For stone textures: stable-diffusion performs best with composite score 0.615 (clip_score=0.302, seam_score=0.829, tiling_score=0.780, fft_score=0.730, stability_score=0.736).

## Category Rankings

### Wood
1. `stable-diffusion` composite=`0.663` seam=`0.919` clip=`0.326`

### Metal
1. `stable-diffusion` composite=`0.587` seam=`0.804` clip=`0.267`

### Stone
1. `stable-diffusion` composite=`0.615` seam=`0.829` clip=`0.302`

## Metric Tables

| Model | Composite | CLIP | Seam | Tiling | FFT | Stability | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable-diffusion | 0.622 | 0.298 | 0.851 | 0.804 | 0.654 | 0.784 | 9 |

## Generated Image Examples

### Wood
Prompt: `seamless wood texture, PBR, high detail, 4k, seamless texture, PBR material`
Texture:

![wood texture](results/experiments/20260429-130504-real-diffusion-run/images/stable-diffusion-wood-000-00.png)
Tiled 2x2:

![wood tiled](results/experiments/20260429-130504-real-diffusion-run/tiles/stable-diffusion-wood-000-00-tile2x2.png)

### Metal
Prompt: `seamless metal surface, industrial, roughness variation, seamless texture, PBR material, high detail`
Texture:

![metal texture](results/experiments/20260429-130504-real-diffusion-run/images/stable-diffusion-metal-001-00.png)
Tiled 2x2:

![metal tiled](results/experiments/20260429-130504-real-diffusion-run/tiles/stable-diffusion-metal-001-00-tile2x2.png)

### Stone
Prompt: `seamless stone texture, natural, high detail, seamless texture, PBR material`
Texture:

![stone texture](results/experiments/20260429-130504-real-diffusion-run/images/stable-diffusion-stone-002-01.png)
Tiled 2x2:

![stone tiled](results/experiments/20260429-130504-real-diffusion-run/tiles/stable-diffusion-stone-002-01-tile2x2.png)

## Analysis

- Texture realism: Mean CLIP alignment is 0.298, mean FFT quality is 0.654, and mean seam continuity is 0.851. Higher CLIP and seam values alongside moderate-to-high FFT values indicate more convincing, production-friendly texture structure.
- Seamless tiling performance: Across the pilot set, tiling consistency averages 0.804 and seam score averages 0.851. The tiled previews help confirm whether numerical seam quality matches visual repetition behavior.
- Prompt sensitivity: The widest within-prompt CLIP spread is 0.074 for `seamless stone texture, natural, high detail, seamless texture, PBR material`, which makes it the most prompt-sensitive case in this pilot.

## Insights

The evaluation pipeline emphasizes tileability and reproducibility because those are often more operationally relevant in graphics production pipelines than visual appeal alone.
When CLIP is available, semantic alignment can be weighted more strongly for concept-sensitive materials. When CLIP is unavailable, the recommendation engine still produces rankings from the remaining metrics.
The procedural baseline is intended to validate infrastructure and should not be treated as a substitute for a learned diffusion model in final R&D comparisons.
