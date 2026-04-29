from __future__ import annotations

import unittest

from gentexlab.prompts import build_prompt_specs, normalize_prompt


class PromptTests(unittest.TestCase):
    def test_normalize_prompt_adds_required_tokens(self) -> None:
        normalized = normalize_prompt("wood texture")
        self.assertIn("seamless texture", normalized)
        self.assertIn("PBR material", normalized)
        self.assertIn("high detail", normalized)

    def test_build_prompt_specs_expands_categories(self) -> None:
        specs = build_prompt_specs(categories=["wood"])
        self.assertGreaterEqual(len(specs), 1)
        self.assertTrue(all(spec.category == "wood" for spec in specs))


if __name__ == "__main__":
    unittest.main()

