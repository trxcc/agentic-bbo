"""Validate submissions by durable receipt identity, not CLI invocation count."""
from __future__ import annotations

import json
from pathlib import Path


def unique_submission_receipts(tool_calls):
    unique = {}
    for call in tool_calls:
        if call.get("tool_name") != "submit_candidate" or not call.get("success"):
            continue
        payload = json.loads(call["result_preview"])
        receipt = payload["result"]
        if not payload.get("ok") or receipt.get("status") != "accepted":
            raise ValueError("Successful submit log has no accepted receipt")
        key = receipt["submission_id"]
        if key in unique:
            previous = json.loads(unique[key]["result_preview"])["result"]
            if previous != receipt:
                raise ValueError(f"Conflicting receipts for {key}")
        else:
            unique[key] = call
    return list(unique.values())


def audit_frontier_run(run_dir: Path, *, initial: int, expected_trials: int):
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    audit = json.loads((run_dir / "tool_setting_audit.json").read_text())
    state = run_dir / "agent_state_archive"
    workspace = run_dir / "workspace_archive"
    if not state.is_dir():
        state = Path(audit["host_state"])
    if not workspace.is_dir():
        workspace = Path(audit["host_workspace"])
    calls = rows(workspace / ".agent_runtime/agent_tool_calls.jsonl")
    submissions = unique_submission_receipts(calls)
    trials = rows(run_dir / "trials.jsonl")
    completed = trials[initial:]
    checks = {
        "exact_budget": len(completed) == len(submissions) == expected_trials,
        "sequential_trial_ids": [t["trial_id"] for t in trials] == list(range(len(trials))),
        "host_request_count": len(list((run_dir / "host_evaluations").glob("request_*.json"))) == expected_trials,
        "host_response_count": len(list((run_dir / "host_evaluations").glob("response_*.json"))) == expected_trials,
    }
    matches = []
    for call, trial in zip(submissions, completed):
        receipt = json.loads(call["result_preview"])["result"]
        candidate = json.loads((state / "context_io" / receipt["context_version"] /
                                (receipt["candidate_id"] + ".json")).read_text())
        request = json.loads((run_dir / "host_evaluations" / f"request_{trial['trial_id']}.json").read_text())
        response = json.loads((run_dir / "host_evaluations" / f"response_{trial['trial_id']}.json").read_text())
        observed, evaluated = trial["objectives"], response["objectives"]
        objective_match = (observed == evaluated or
                           (len(observed) == len(evaluated) == 1 and
                            next(iter(observed.values())) == next(iter(evaluated.values()))))
        matches.append(candidate["config"] == trial["config"] == request["config"]
                       and trial["status"] == response["status"] == "success"
                       and objective_match)
    checks["candidate_host_trial_match"] = len(matches) == expected_trials and all(matches)
    total_calls = sum(c.get("tool_name") == "submit_candidate" and c.get("success", False) for c in calls)
    return {"run_dir": str(run_dir), "passed": all(checks.values()), "checks": checks,
            "successful_submit_calls": total_calls, "unique_submissions": len(submissions),
            "idempotent_repeats": total_calls-len(submissions), "new_trials": len(completed)}
