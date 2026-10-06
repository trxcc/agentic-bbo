"""The CLI must propagate structured tool failures to shell control flow."""
import asyncio
import json
import os
import subprocess
import sys

import pytest

from bbo.algorithms.agentic.general_agent_engines import (
    _HOST_TOOL_CLIENT,
    _start_host_tool_server,
)


@pytest.mark.parametrize("output,expected", [
    ('{"ok":false,"error":"invalid_candidate"}', 2),
    ({"ok": False, "error": "unknown_field"}, 2),
    ('{"ok":true,"candidate_id":"candidate_1"}', 0),
    ({"ok": True}, 0),
    ({"parameters": ["x"]}, 0),
    ('plain text containing "ok":false', 0),
    ('[1,2]', 0),
])
def test_tool_result_controls_exit_status_and_shell_chaining(tmp_path, output, expected):
    async def scenario():
        async def execute(name, arguments, call_id):
            return output

        client = tmp_path / "bbo_tool.py"
        client.write_text(_HOST_TOOL_CLIENT)
        marker = tmp_path / "continued"
        server, thread = _start_host_tool_server(
            allowed={"write_candidate"}, executor=execute,
            loop=asyncio.get_running_loop(), max_calls=1,
        )
        env = {**os.environ,
               "BBO_HOST_TOOL_SOCKET": f"tcp://127.0.0.1:{server.server_address[1]}",
               "BBO_HOST_TOOL_DEADLINE": "0"}
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", '"$1" "$2" write_candidate \'{}\' && touch "$3"',
                 "test", sys.executable, str(client), str(marker)],
                env=env, capture_output=True, text=True, timeout=10,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        assert completed.returncode == expected
        assert marker.exists() is (expected == 0)
        assert completed.stdout.strip() == (
            output if isinstance(output, str) else json.dumps(output, sort_keys=True))

    asyncio.run(scenario())
