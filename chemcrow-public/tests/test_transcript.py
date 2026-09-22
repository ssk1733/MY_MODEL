from datetime import datetime
from uuid import UUID

from langchain.schema import AgentAction

from chemcrow.agents.transcript import TranscriptCallbackHandler


def test_transcript_saves_complete_agent_run(tmp_path):
    times = iter(
        [
            datetime(2026, 9, 10, 15, 25, 1),
            datetime(2026, 9, 10, 15, 25, 9),
        ]
    )
    handler = TranscriptCallbackHandler(
        result_dir=str(tmp_path),
        now_factory=lambda: next(times),
    )
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    action = AgentAction(
        tool="MooRetroSingleStepPredict",
        tool_input='{"target_smiles":"CCO"}',
        log="Thought: call local model\nAction: MooRetroSingleStepPredict",
    )

    handler.on_chain_start({}, {"input": "预测 CCO"}, run_id=run_id)
    handler.on_chain_end(
        {
            "output": "最终答案",
            "intermediate_steps": [(action, '{"ok": true}')],
        },
        run_id=run_id,
    )

    output_path = tmp_path / "202609101525.txt"
    content = output_path.read_text(encoding="utf-8")
    assert "预测 CCO" in content
    assert "MooRetroSingleStepPredict" in content
    assert '{"target_smiles":"CCO"}' in content
    assert '{"ok": true}' in content
    assert "最终答案" in content
    assert handler.last_path == str(output_path)


def test_transcript_appends_runs_from_the_same_minute(tmp_path):
    times = iter(
        [
            datetime(2026, 9, 10, 15, 25, 1),
            datetime(2026, 9, 10, 15, 25, 2),
            datetime(2026, 9, 10, 15, 25, 3),
            datetime(2026, 9, 10, 15, 25, 4),
        ]
    )
    handler = TranscriptCallbackHandler(
        result_dir=str(tmp_path),
        now_factory=lambda: next(times),
    )

    for suffix in (1, 2):
        run_id = UUID(f"00000000-0000-0000-0000-{suffix:012d}")
        handler.on_chain_start({}, {"input": f"question {suffix}"}, run_id=run_id)
        handler.on_chain_end(
            {"output": f"answer {suffix}", "intermediate_steps": []},
            run_id=run_id,
        )

    content = (tmp_path / "202609101525.txt").read_text(encoding="utf-8")
    assert "question 1" in content
    assert "answer 1" in content
    assert "question 2" in content
    assert "answer 2" in content
    assert content.count("ChemCrow run ID:") == 2
