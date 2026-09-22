import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase
from chemcrow.tools import *

def make_tools(
    llm: BaseLanguageModel,
    api_keys: dict = None,
    local_rxn: bool = False,
    verbose: bool = True,
    enzyme_planner_config: dict = None,
):
    api_keys = dict(api_keys or {})
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    structured_database = EnzymeReactionDatabase()
    search_model = EnzRetroClient(single_fn_type="biocyc")

    all_tools = agents.load_tools(
        [
            "python_repl",
            # "ddg-search",

            # 联网工具，访问 Wikipedia
            # "wikipedia",

            # "human"
        ]
    )

    all_tools += [
        MoleculeInputNormalizer(database_service=structured_database),
        Query2CAS(),
        SMILES2Name(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
        SafetySummary(llm=llm),
        Scholar2ResultLLM(
            llm=llm,
            openai_api_key=openai_api_key,
            semantic_scholar_api_key=semantic_scholar_api_key
        ),
        EnzymeRetroDatabase(),
        # Direct one-step prediction remains independently available.
        MooRetroSingleStepPredict(single_fn_type="biocyc"),
        EnzymeSingleStepEvaluator(
            database_service=structured_database,
            config=enzyme_planner_config,
        ),
        # Search owns graph traversal; the LLM never has to maintain tree state.
        EnzymeRetrosynthesisSearch(
            database_service=structured_database,
            model_client=search_model,
            config=enzyme_planner_config,
        ),
        EnzymeMultiStepEvaluator(),
    ]
    if chemspace_api_key:
        all_tools += [GetMoleculePrice(chemspace_api_key)]
    if serp_api_key:
        all_tools += [WebSearch(serp_api_key)]
    if (not local_rxn) and rxn4chem_api_key:
        all_tools += [
            RXNPredict(rxn4chem_api_key),
            RXNRetrosynthesis(rxn4chem_api_key, openai_api_key),
        ]
    elif local_rxn:
        all_tools += [
            RXNPredictLocal(),
            RXNRetrosynthesisLocal(),
        ]

    return all_tools
