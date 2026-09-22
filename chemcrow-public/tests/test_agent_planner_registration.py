import unittest

from langchain.llms.fake import FakeListLLM

from chemcrow.agents.tools import make_tools


class AgentPlannerRegistrationTest(unittest.TestCase):
    def test_make_tools_registers_layered_retrosynthesis_workflow(self):
        tools = make_tools(FakeListLLM(responses=["ok"]))
        names = [tool.name for tool in tools]
        expected = (
            "MoleculeInputNormalizer",
            "EnzymeRetroDatabase",
            "MooRetroSingleStepPredict",
            "EnzymeSingleStepEvaluator",
            "EnzymeRetrosynthesisSearch",
            "EnzymeMultiStepEvaluator",
        )
        for name in expected:
            self.assertIn(name, names)
            self.assertEqual(names.count(name), 1)
        self.assertNotIn("Name2SMILES", names)
        self.assertNotIn("EnzymeRetroPathFinder", names)
        self.assertNotIn("HybridEnzymeRetroPlanner", names)

    def test_make_tools_does_not_mutate_api_key_mapping(self):
        api_keys = {"OPENAI_API_KEY": "test"}
        original = dict(api_keys)

        make_tools(FakeListLLM(responses=["ok"]), api_keys=api_keys)

        self.assertEqual(api_keys, original)


if __name__ == "__main__":
    unittest.main()
