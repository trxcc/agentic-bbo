from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bbo.algorithms.agentic.round_protocol import AgentRoundState, RoundStage
from bbo.algorithms.agentic.general_agent_engines import (
    _HOST_TOOL_CLIENT,
    _start_host_tool_server,
)


BASE_SPECS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": ({"candidate": {"type": "object"}} if name == "validate_candidate" else {}),
            },
        },
    }
    for name in (
        "get_trial_history",
        "assess_backend_suitability",
        "optimizer_suggest",
        "validate_candidate",
    )
]


def tool_names(state: AgentRoundState) -> set[str]:
    return {spec["function"]["name"] for spec in state.tool_specs()}


def transport_tool_names(state: AgentRoundState) -> set[str]:
    return {
        spec["function"]["name"] for spec in state.transport_tool_specs()
    }


def decode(raw: str) -> dict:
    return json.loads(raw)


@pytest.fixture
def state(tmp_path: Path) -> AgentRoundState:
    history = [SimpleNamespace(suggestion=SimpleNamespace(trial_id=19))]
    return AgentRoundState(
        round_id="round_00001",
        history=history,
        base_specs=BASE_SPECS,
        event_path=tmp_path / "round_events.jsonl",
    )


async def base_executor(name: str, arguments: dict, _call_id: str | None) -> str:
    if name == "get_trial_history":
        result = {"trials": []}
    elif name == "assess_backend_suitability":
        result = {"gp_ei": {"status": "ok"}, "tpe": {"status": "usable"}}
    elif name == "optimizer_suggest":
        result = {"backend": arguments["backend"], "candidate": {"x": 0.25}, "identity": "proposal_identity"}
    elif name == "validate_candidate":
        candidate = arguments["candidate"]
        result = {"valid": True, "config": candidate, "identity": f"valid_{candidate['x']}"}
    else:  # pragma: no cover
        raise AssertionError(name)
    return json.dumps({"ok": True, "result": result})


def test_round_protocol_is_single_transcript_state_gated(state: AgentRoundState) -> None:
    async def run() -> None:
        await _assert_single_transcript(state)

    asyncio.run(run())


def test_static_transport_advertises_complete_lifecycle(
    state: AgentRoundState,
) -> None:
    expected = {
        "get_trial_history",
        "assess_backend_suitability",
        "optimizer_suggest",
        "validate_candidate",
        "commit_candidate",
    }
    assert transport_tool_names(state) == expected
    assert tool_names(state) == {
        "get_trial_history",
        "assess_backend_suitability",
    }


def test_codex_cli_transport_completes_state_gated_round(
    state: AgentRoundState, tmp_path: Path
) -> None:
    async def scenario() -> None:
        async def execute(
            name: str, arguments: dict, call_id: str | None
        ) -> str:
            return await state.execute(base_executor, name, arguments, call_id)

        client = tmp_path / "bbo_tool.py"
        client.write_text(_HOST_TOOL_CLIENT, encoding="utf-8")
        allowed = transport_tool_names(state)
        server, thread = _start_host_tool_server(
            allowed=allowed,
            executor=execute,
            loop=asyncio.get_running_loop(),
            max_calls=16,
        )
        env = {
            **os.environ,
            "BBO_HOST_TOOL_SOCKET": (
                f"tcp://127.0.0.1:{server.server_address[1]}"
            ),
            "BBO_AGENT_CALL_ID": "codex-cli-round",
        }

        async def invoke(name: str, arguments: dict) -> dict:
            completed = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(client), name, json.dumps(arguments)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            return json.loads(completed.stdout)

        try:
            assessed = await invoke("assess_backend_suitability", {})
            assert assessed["result"]["evidence_id"].startswith("evidence_")
            suggested = await invoke("optimizer_suggest", {"backend": "gp_ei"})
            proposal_id = suggested["result"]["proposal_id"]
            validated = await invoke(
                "validate_candidate",
                {"proposal_id": proposal_id, "candidate": {"x": 0.25}},
            )
            committed = await invoke(
                "commit_candidate",
                {
                    "validation_token": validated["result"]["validation_token"],
                    "relationship": "adopt",
                    "backend_rationale": "GP evidence is adequate.",
                    "modification_rationale": "Adopt the exact proposal.",
                    "belief": "The response is locally smooth.",
                    "expected_information": "Test the local basin.",
                    "hypothesis": "The proposal improves the loss.",
                    "hypothesis_update": {
                        "status": "inconclusive",
                        "reason": "The previous observation was weak evidence.",
                    },
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert committed["result"]["terminal"] is True
        assert committed["result"]["candidate_payload"]["candidates"][0][
            "config"
        ] == {"x": 0.25}
        assert state.stage is RoundStage.COMPLETED

    asyncio.run(scenario())


async def _assert_single_transcript(state: AgentRoundState) -> None:
    assert state.stage is RoundStage.COLLECT_EVIDENCE
    assert tool_names(state) == {"get_trial_history", "assess_backend_suitability"}

    out_of_order = decode(
        await state.execute(base_executor, "optimizer_suggest", {"backend": "gp_ei"}, "bad")
    )
    assert out_of_order["error"] == "protocol_error"
    assert state.stage is RoundStage.COLLECT_EVIDENCE

    assessed = decode(
        await state.execute(base_executor, "assess_backend_suitability", {}, "assess")
    )
    evidence_id = assessed["result"]["evidence_id"]
    assert evidence_id.startswith("evidence_")
    assert state.stage is RoundStage.SELECT_BACKEND
    assert tool_names(state) == {"get_trial_history", "optimizer_suggest"}

    assessed_replay = decode(
        await state.execute(base_executor, "assess_backend_suitability", {}, "assess-duplicate")
    )
    assert assessed_replay["result"]["evidence_id"] == evidence_id

    suggested = decode(
        await state.execute(base_executor, "optimizer_suggest", {"backend": "gp_ei"}, "suggest")
    )
    proposal_id = suggested["result"]["proposal_id"]
    assert suggested["result"]["evidence_id"] == evidence_id
    assert state.stage is RoundStage.VALIDATE
    suggested_replay = decode(
        await state.execute(base_executor, "optimizer_suggest", {"backend": "gp_ei"}, "suggest-duplicate")
    )
    assert suggested_replay["result"]["proposal_id"] == proposal_id

    validated = decode(
        await state.execute(
            base_executor,
            "validate_candidate",
            {"proposal_id": proposal_id, "candidate": {"x": 0.2}},
            "validate",
        )
    )
    validation_token = validated["result"]["validation_token"]
    assert state.stage is RoundStage.COMMIT
    assert "commit_candidate" in tool_names(state)

    committed = decode(
        await state.execute(
            base_executor,
            "commit_candidate",
            {
                "validation_token": validation_token,
                "relationship": "refine",
                "backend_rationale": "GP evidence is adequate.",
                "modification_rationale": "Move slightly toward the incumbent.",
                "belief": "The response is locally smooth.",
                "expected_information": "Test the local basin.",
                "hypothesis": "The revised point improves the loss.",
                "hypothesis_update": {"status": "inconclusive", "reason": "The previous result only weakly changed the belief."},
            },
            "commit",
        )
    )
    assert committed["result"]["terminal"] is True
    assert state.stage is RoundStage.COMPLETED
    candidate = committed["result"]["candidate_payload"]["candidates"][0]
    assert candidate["config"] == {"x": 0.2}
    assert candidate["search_action"]["hypothesis_update"]["evidence_trial_id"] == 19
    assert candidate["search_action"]["optimizer"]["candidate_diff"] == [
        {"name": "x", "proposal": 0.25, "final": 0.2}
    ]


def test_round_protocol_rejects_stale_tokens_and_false_adoption(state: AgentRoundState) -> None:
    async def run() -> None:
        await _assert_rejections(state)

    asyncio.run(run())


async def _assert_rejections(state: AgentRoundState) -> None:
    await state.execute(base_executor, "assess_backend_suitability", {}, "assess")
    suggested = decode(
        await state.execute(base_executor, "optimizer_suggest", {"backend": "tpe"}, "suggest")
    )
    proposal_id = suggested["result"]["proposal_id"]
    validated = decode(
        await state.execute(
            base_executor,
            "validate_candidate",
            {"proposal_id": proposal_id, "candidate": {"x": 0.1}},
            "validate",
        )
    )
    common = {
        "relationship": "adopt",
        "backend_rationale": "TPE has stronger separation.",
        "modification_rationale": "Use the proposal.",
        "belief": "The elite region is separated.",
        "expected_information": "Test the elite region.",
        "hypothesis": "The point improves loss.",
        "hypothesis_update": {"status": "supported", "reason": "The last observation improved."},
    }
    stale = decode(
        await state.execute(
            base_executor,
            "commit_candidate",
            {**common, "validation_token": "stale"},
            "bad-token",
        )
    )
    assert stale["error"] == "protocol_error"

    false_adopt = decode(
        await state.execute(
            base_executor,
            "commit_candidate",
            {**common, "validation_token": validated["result"]["validation_token"]},
            "false-adopt",
        )
    )
    assert false_adopt["error"] == "protocol_error"
    assert "differs" in false_adopt["message"]
    assert state.stage is RoundStage.COMMIT
