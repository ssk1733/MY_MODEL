import json
from types import SimpleNamespace

from chemcrow.tools.mooretro import MooRetroSingleStepPredict
from chemcrow.planner import EnzRetroClient
from chemcrow.planner.chem_utils import extract_ec_numbers
from chemcrow.planner.models import MoleculeNode


def test_mooretro_plain_smiles_returns_json(monkeypatch):
    tool = MooRetroSingleStepPredict(max_results=1)
    prediction = SimpleNamespace(
        to_dict={
            "reactants": "CCO",
            "ec": "1.1.1.1",
            "scores": 0.9,
            "templates": "template",
        }
    )
    monkeypatch.setattr(tool, "_single_fn", lambda **kwargs: [prediction])

    result = json.loads(tool.run("CC=O"))

    assert result["ok"] is True
    assert result["prediction_count"] == 1
    assert result["predictions"][0]["reactants"] == "CCO"


def test_mooretro_json_passes_building_blocks(monkeypatch):
    tool = MooRetroSingleStepPredict()
    captured = {}

    def fake_single_fn(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(tool, "_single_fn", fake_single_fn)
    result = json.loads(
        tool.run(
            '{"target_smiles":"CC=O","building_blocks":["CCO","O"]}'
        )
    )

    assert result["ok"] is True
    assert captured["target_smile"] == "CC=O"
    assert captured["building_blocks"] == ["CCO", "O"]
    assert captured["single_fn_type"] == "biocyc"
    assert captured["ec_level"] == 3


def test_model_alias_selects_neural_biocyc_backend(monkeypatch):
    tool = MooRetroSingleStepPredict(single_fn_type="model")
    captured = {}

    def fake_single_fn(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(tool, "_single_fn", fake_single_fn)
    result = json.loads(tool.run("CCO"))

    assert result["ok"] is True
    assert result["model"] == "biocyc"
    assert result["backend_mode"] == "neural_model"
    assert captured["single_fn_type"] == "biocyc"


def test_xlsx_backend_is_reported_as_database_lookup(monkeypatch):
    tool = MooRetroSingleStepPredict(single_fn_type="model_xlsx")
    monkeypatch.setattr(tool, "_single_fn", lambda **kwargs: [])

    result = json.loads(tool.run("CCO"))

    assert result["ok"] is True
    assert result["backend_mode"] == "database_lookup"
    assert result["runtime_device"] == "cpu"

def test_mooretro_canonicalizes_input_and_predicted_reactants(monkeypatch):
    tool = MooRetroSingleStepPredict(max_results=2)
    captured = {}

    def fake_single_fn(**kwargs):
        captured.update(kwargs)
        return [
            SimpleNamespace(to_dict={"reactants": "OCC.C(O)", "scores": 0.9}),
            SimpleNamespace(to_dict={"reactants": "not a smiles", "scores": 0.8}),
        ]

    monkeypatch.setattr(tool, "_single_fn", fake_single_fn)
    result = json.loads(
        tool.run(
            json.dumps(
                {
                    "target_smiles": {"canonical_smiles": "OCC"},
                    "building_blocks": [{"canonical_smiles": "C(O)"}],
                }
            )
        )
    )

    assert result["ok"] is True
    assert captured["target_smile"] == "CCO"
    assert captured["building_blocks"] == ["CO"]
    assert result["input_normalization"] == "RDKIT_CANONICAL_ISOMERIC_SMILES"
    assert result["prediction_count"] == 1
    assert result["rejected_prediction_count"] == 1
    prediction = result["predictions"][0]
    assert prediction["raw_reactants"] == "OCC.C(O)"
    assert prediction["reactant_smiles"] == ["CCO", "CO"]
    assert prediction["reactants"] == "CCO.CO"


def test_ec_parser_accepts_three_and_four_level_ec_numbers():
    assert extract_ec_numbers("[EC]2.6.1") == ("2.6.1",)
    assert extract_ec_numbers("EC 1.1.1.1") == ("1.1.1.1",)


def test_model_client_reads_ec_from_mooretro_template(tmp_path):
    client = EnzRetroClient(
        moo_root=str(tmp_path),
        predictor=lambda **kwargs: [
            {"reactants": "CCO", "templates": "[EC]2.6.1", "scores": 0.9}
        ],
    )
    target = MoleculeNode(
        key="SMI:CC=O",
        query="CC=O",
        smiles="CC=O",
        canonical_smiles="CC=O",
    )
    candidates = client.predict_candidates(target, [], top_k=1)
    assert len(candidates) == 1
    assert candidates[0].ec_numbers == ("2.6.1",)



def test_model_client_preserves_all_template_precursors(tmp_path):
    client = EnzRetroClient(
        moo_root=str(tmp_path),
        predictor=lambda **kwargs: [
            {
                "reactants": "CCO",
                "all_reactants": ["CCO", "O"],
                "templates": "[EC]1.1.1.1",
                "scores": 0.9,
            }
        ],
    )
    target = MoleculeNode(
        key="SMI:CC=O",
        query="CC=O",
        smiles="CC=O",
        canonical_smiles="CC=O",
    )
    candidates = client.predict_candidates(target, [], top_k=1)
    assert len(candidates) == 1
    assert {item.canonical_smiles for item in candidates[0].precursors} == {"CCO", "O"}


def test_model_client_model_alias_is_neural_backend(tmp_path):
    client = EnzRetroClient(
        moo_root=str(tmp_path), single_fn_type="model", predictor=lambda **kwargs: []
    )

    info = client.cache_info()

    assert client.single_fn_type == "biocyc"
    assert info["backend_mode"] == "neural_model"

def test_mooretro_rejects_invalid_smiles_before_loading_model():
    tool = MooRetroSingleStepPredict()
    result = json.loads(tool.run("not a smiles"))

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_smiles"
