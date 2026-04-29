from __future__ import annotations

from dataclasses import dataclass


REQUIRED_TOKENS = ("seamless texture", "PBR material", "high detail")

PROMPT_LIBRARY: dict[str, list[str]] = {
    "wood": [
        "oak grain with subtle knots",
        "weathered cedar planks",
        "polished walnut veneer",
    ],
    "metal": [
        "brushed aluminum surface",
        "oxidized steel plate",
        "machined titanium panel",
    ],
    "fabric": [
        "woven canvas material",
        "soft wool weave",
        "technical nylon mesh",
    ],
    "stone": [
        "granite slab with natural speckles",
        "aged sandstone surface",
        "slate rock pattern",
    ],
    "plastic": [
        "matte injection-molded polymer",
        "glossy industrial ABS shell",
        "recycled plastic composite",
    ],
}


@dataclass(slots=True)
class PromptSpec:
    raw_prompt: str
    structured_prompt: str
    category: str


def normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip()
    lower = normalized.lower()
    missing = [token for token in REQUIRED_TOKENS if token.lower() not in lower]
    if missing:
        normalized = ", ".join([normalized, *missing])
    return normalized


def infer_category(prompt: str) -> str:
    lowered = prompt.lower()
    for category in PROMPT_LIBRARY:
        if category in lowered:
            return category
    keyword_map = {
        "oak": "wood",
        "cedar": "wood",
        "grain": "wood",
        "steel": "metal",
        "aluminum": "metal",
        "fabric": "fabric",
        "cloth": "fabric",
        "stone": "stone",
        "granite": "stone",
        "plastic": "plastic",
        "polymer": "plastic",
    }
    for keyword, category in keyword_map.items():
        if keyword in lowered:
            return category
    return "generic"


def build_prompt_specs(
    prompts: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[PromptSpec]:
    prompt_specs: list[PromptSpec] = []

    if prompts:
        for prompt in prompts:
            prompt_specs.append(
                PromptSpec(
                    raw_prompt=prompt,
                    structured_prompt=normalize_prompt(prompt),
                    category=infer_category(prompt),
                )
            )

    if categories:
        for category in categories:
            descriptors = PROMPT_LIBRARY.get(category.lower(), [f"{category} material surface"])
            for descriptor in descriptors:
                raw_prompt = f"{category} texture, {descriptor}"
                prompt_specs.append(
                    PromptSpec(
                        raw_prompt=raw_prompt,
                        structured_prompt=normalize_prompt(raw_prompt),
                        category=category.lower(),
                    )
                )

    if not prompt_specs:
        for category, descriptors in PROMPT_LIBRARY.items():
            descriptor = descriptors[0]
            raw_prompt = f"{category} texture, {descriptor}"
            prompt_specs.append(
                PromptSpec(
                    raw_prompt=raw_prompt,
                    structured_prompt=normalize_prompt(raw_prompt),
                    category=category,
                )
            )

    deduplicated: list[PromptSpec] = []
    seen: set[tuple[str, str]] = set()
    for spec in prompt_specs:
        key = (spec.structured_prompt, spec.category)
        if key not in seen:
            seen.add(key)
            deduplicated.append(spec)
    return deduplicated

