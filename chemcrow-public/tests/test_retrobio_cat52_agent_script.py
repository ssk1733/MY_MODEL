from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrobio_cat52.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retrobio_cat52", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TargetSubstratesAgentHarnessTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "benchmark_id": "TEST",
            "input": {
                "target_smiles": "CCN",
                "substrate_smiles": ["CC=O", "N"],
            },
            "expected": {"enzyme_types": ["TA", "ADH"]},
        }
        self.tool_result = {
            "ok": True,
            "routes": [
                {
                    "rank": 1,
                    "complete": True,
                    "enzymes": ["ADH", "TA"],
                    "steps": [
                        {
                            "target": {"key": "target", "canonical_smiles": "CCN"},
                            "precursors": [
                                {"key": "a", "canonical_smiles": "CC=O"},
                                {"key": "b", "canonical_smiles": "N"},
                            ],
                        }
                    ],
                }
            ],
            "stats": {},
        }

    def _agent(
        self, building_blocks=None, *, use_model=True, include_normalizer=True
    ):
        blocks = building_blocks or ["N", "CC=O"]
        normalization_action = type(
            "NormalizationAction",
            (),
            {
                "tool": "MoleculeInputNormalizer",
                "tool_input": json.dumps({
                    "targets": ["CCN"],
                    "building_blocks": blocks,
                }),
            },
        )()
        normalization_result = {
            "ok": True,
            "normalized_input": {
                "target": "CCN",
                "targets": ["CCN"],
                "building_blocks": blocks,
            },
        }
        action = type(
            "SearchAction",
            (),
            {
                "tool": "EnzymeRetrosynthesisSearch",
                "tool_input": json.dumps({
                    "targets": ["CCN"],
                    "building_blocks": blocks,
                    "endpoint_mode": "exact_set",
                    "top_k": 10,
                    "search_algorithm": "and_or_mcts",
                    "config": {
                        "use_database": True,
                        "use_model": use_model,
                    },
                }),
            },
        )()
        tool_result = self.tool_result

        class Executor:
            def __call__(self, payload):
                steps = []
                if include_normalizer:
                    steps.append(
                        (normalization_action, json.dumps(normalization_result))
                    )
                steps.append((action, json.dumps(tool_result)))
                return {
                    "output": json.dumps(tool_result),
                    "intermediate_steps": steps,
                }

        return type("Agent", (), {"agent_executor": Executor()})()

    def test_prompt_contains_target_and_substrates_but_not_gold_enzymes(self):
        prompt = MODULE.build_agent_prompt("CCN", ["CC=O", "N"], 10, 4)
        self.assertIn("CCN", prompt)
        self.assertIn("CC=O", prompt)
        self.assertNotIn('"TA"', prompt)
        self.assertIn('"targets": ["CCN"]', prompt)
        self.assertIn('"endpoint_mode": "exact_set"', prompt)
        self.assertIn('"use_database": true', prompt)
        self.assertIn('"use_model": true', prompt)
        self.assertIn("叶节点集合必须与 D 完全相等", prompt)

    def test_agent_result_uses_exact_search_tool_call_and_routes(self):
        result = MODULE.run_agent_case(
            self._agent(), self.case, top_k=10, max_depth=4
        )
        self.assertEqual(result["prediction"]["route_count"], 1)
        self.assertEqual(
            result["prediction"]["stats"]["valid_normalization_call_count"], 1
        )
        self.assertEqual(result["search_result"], self.tool_result)
        payload = json.loads(result["tool_calls"][-1]["tool_input"])
        self.assertEqual(payload["targets"], ["CCN"])
        self.assertEqual(payload["endpoint_mode"], "exact_set")
        evaluated = MODULE.evaluate_case_result(self.case, result["prediction"])
        self.assertEqual(evaluated["first_hit_rank"]["complete_route_found"], 1)
        self.assertEqual(evaluated["first_hit_rank"]["enzyme_all_present"], 1)

    def test_agent_retries_after_protocol_error(self):
        valid_agent = self._agent()
        valid_executor = valid_agent.agent_executor

        class RetryExecutor:
            def __init__(self):
                self.calls = 0

            def __call__(self, payload):
                self.calls += 1
                if self.calls == 1:
                    return {"output": "no tool", "intermediate_steps": []}
                return valid_executor(payload)

        executor = RetryExecutor()
        agent = type("Agent", (), {"agent_executor": executor})()
        result = MODULE.run_agent_case(
            agent, self.case, top_k=10, max_depth=4, agent_attempts=2
        )
        self.assertEqual(executor.calls, 2)
        self.assertEqual(result["prediction"]["stats"]["agent_attempt_count"], 2)
        self.assertEqual(len(result["attempts"]), 2)

    def test_wrong_or_incomplete_building_block_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "targets/building_blocks"):
            MODULE.run_agent_case(
                self._agent(["CC=O"]), self.case, top_k=10, max_depth=4
            )

    def test_model_and_database_must_both_be_enabled(self):
        with self.assertRaisesRegex(ValueError, "use_database=true and use_model=true"):
            MODULE.run_agent_case(
                self._agent(use_model=False), self.case, top_k=10, max_depth=4
            )

    def test_normalization_must_precede_search(self):
        with self.assertRaisesRegex(ValueError, "normalize the exact"):
            MODULE.run_agent_case(
                self._agent(include_normalizer=False),
                self.case,
                top_k=10,
                max_depth=4,
            )

    def test_current_route_bundle_output_can_be_evaluated(self):
        bundled = {
            "ok": True,
            "mode": "multi-target",
            "routes": [
                {
                    "rank": 1,
                    "target_routes": [
                        {
                            "target": {"canonical_smiles": "CCN"},
                            "route_rank": 1,
                            "route": self.tool_result["routes"][0],
                        }
                    ],
                }
            ],
        }
        prediction = MODULE._normalize_prediction(bundled, 10, "CCN")
        self.assertEqual(prediction["route_count"], 1)
        self.assertEqual(prediction["routes"][0]["steps"][0]["target"]["key"], "target")


if __name__ == "__main__":
    unittest.main()
