from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agent_retrosynthesis.py"
SPEC = importlib.util.spec_from_file_location("run_agent_retrosynthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgentRetrosynthesisCLITest(unittest.TestCase):
    def test_prompt_preserves_multiple_targets_stock_and_search_policy(self):
        prompt = MODULE.build_prompt(
            ["CCO", "CCN"],
            ["O", "N"],
            "/tmp/stock.smi",
            "any_stock",
            7,
            9,
        )
        self.assertIn('"targets": ["CCO", "CCN"]', prompt)
        self.assertIn('"building_blocks": ["O", "N"]', prompt)
        self.assertIn('"building_block_library": "/tmp/stock.smi"', prompt)
        self.assertIn('"endpoint_mode": "any_stock"', prompt)
        self.assertIn('"top_k": 7', prompt)
        self.assertIn('"use_database": true', prompt)
        self.assertIn('"use_model": true', prompt)
        self.assertIn("AND-OR", prompt)

    def test_deepseek_api_base_has_official_default_and_allows_override(self):
        self.assertEqual(MODULE.DEFAULT_DEEPSEEK_MODEL, "deepseek-flash")
        self.assertEqual(
            MODULE.resolve_api_base("deepseek-flash", None),
            "https://api.deepseek.com",
        )
        self.assertEqual(
            MODULE.resolve_api_base("deepseek-flash", "https://proxy.example/v1/"),
            "https://proxy.example/v1",
        )
        self.assertIsNone(MODULE.resolve_api_base("gpt-4o", None))

    def test_observation_parser_accepts_mapping_and_json(self):
        self.assertEqual(MODULE._observation_object({"ok": True}), {"ok": True})
        self.assertEqual(
            MODULE._observation_object(json.dumps({"ok": True})), {"ok": True}
        )
        self.assertIsNone(MODULE._observation_object("not json"))


if __name__ == "__main__":
    unittest.main()
