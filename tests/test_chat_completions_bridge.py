"""Generic Chat Completions must preserve native tool sessions without vendor flags."""
import json
import pytest

from bbo.algorithms.agentic.codex_responses_compat import (
    ChatCompletionAccumulator, responses_to_sglang_chat_request,
)
from bbo.algorithms.agentic.raw_agentic_bbo import create_raw_agentic_bbo


def test_generic_bridge_encodes_schema_and_replayed_call_consistently():
    request = {
        "model": "vendor/model", "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": "inspect"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Inspect first."}]},
            {"type": "function_call", "namespace": "native", "name": "inspect", "call_id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ready"},
        ],
        "tools": [{"type": "namespace", "name": "native", "tools": [
            {"type": "function", "name": "inspect", "parameters": {"type": "object"}},
        ]}],
    }
    result = json.loads(responses_to_sglang_chat_request(json.dumps(request).encode(), dialect="chat_completions"))
    assert not set(result).intersection({"thinking", "reasoning_effort", "separate_reasoning", "chat_template_kwargs"})
    assert result["tools"][0]["function"]["name"] == "native__inspect"
    assert result["messages"][1]["tool_calls"][0]["function"]["name"] == "native__inspect"
    assert result["messages"][1]["reasoning_content"] == "Inspect first."
    assert result["messages"][2]["tool_call_id"] == "c1"
    assert result["stream_options"] == {"include_usage": True}


def test_generic_stream_restores_namespace_and_preserves_usage():
    accumulator = ChatCompletionAccumulator({"model": "vendor/model"}, dialect="chat_completions")
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Inspect first."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "native__inspect", "arguments": '{"x":'}}]}}]},
        {"choices": [{"finish_reason": "tool_calls", "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens_details": {"reasoning_tokens": 5}}},
    ]
    for chunk in chunks:
        accumulator.add(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    assert accumulator.add(b"data: [DONE]\n\n")
    events = [json.loads(next(line[5:] for line in event.decode().splitlines() if line.startswith("data:"))) for event in accumulator.events()]
    final = events[-1]["response"]
    call = next(item for item in final["output"] if item["type"] == "function_call")
    assert (call["namespace"], call["name"], call["call_id"], call["arguments"]) == ("native", "inspect", "c1", '{"x":1}')
    assert final["usage"]["total_tokens"] == 112
    assert final["usage"]["input_tokens_details"]["cached_tokens"] == 20
    assert final["usage"]["output_tokens_details"]["reasoning_tokens"] == 5


@pytest.mark.parametrize("provider", ["sinapisai", "voyageage"])
def test_gateway_native_config_selects_generic_bridge(tmp_path, provider):
    agent = create_raw_agentic_bbo(framework="codex", provider=provider, model="vendor/model",
        api_base=f"https://api.{provider}.com/v1", api_key_env="GATEWAY_API_KEY",
        context_access="on_demand", tool_mode="function_calling", max_tool_calls=0,
        execution_backend="isolated_docker", enable_memory=False, run_dir=tmp_path)
    config = agent._codex_config()
    assert config["responses_api_compat"] == "chat_completions"
    assert config["black_box_required"]
    assert config["execution_backend"] == "isolated_docker"
    assert config["native_round_guard"] is True
    assert agent.config.max_tool_calls == 0


def test_historical_native_run_cannot_silently_resume_with_new_guard(tmp_path):
    import pytest
    from bbo.algorithms import create_algorithm
    from bbo.algorithms.agentic.general_agent_engines import MockAgentEngine
    from conftest import create_agent_test_task

    def algorithm(resume=False):
        return create_algorithm('raw_agentic_bbo', engine=MockAgentEngine(), framework='codex',
            provider='sinapisai',model='offline',api_base='http://127.0.0.1:1/v1',
            context_access='on_demand',tool_mode='function_calling',
            run_dir=tmp_path/'run',resume=resume)
    task=create_agent_test_task(max_evaluations=4,seed=1)
    agent=algorithm()
    agent.setup(task.spec,seed=1,task_description=task.get_description())
    agent._persist_state()
    path=agent._agent_state_path
    state=json.loads(path.read_text())
    assert state['native_round_guard_version']=='native_round_guard_v1'
    same=algorithm(resume=True)
    same.setup(task.spec,seed=1,task_description=task.get_description())
    state.pop('native_round_guard_version')
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='different native round guard'):
        algorithm(resume=True).setup(task.spec,seed=1,task_description=task.get_description())
