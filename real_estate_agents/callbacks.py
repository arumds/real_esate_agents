"""
real_estate_agents/callbacks.py

ADK callback hooks, two different patterns:

  - Logging (before/after_tool_log, before/after_model_log) always returns
    None. It never changes what the agent/tool/model actually does -- it
    just records latency (and token counts, for model calls) into CALL_LOG
    as a side effect. Safe to attach anywhere.

  - The guardrail (validate_valuation_input) returns a value when it
    decides to intervene. In ADK, a before_tool_callback's non-None return
    is used AS the tool's response INSTEAD OF running the real tool -- the
    same short-circuit mechanism you'd use for cost limits or caching, just
    applied to input validation here.

Wired onto the two LlmAgents in real_estate_agents/agent.py.
"""

from __future__ import annotations

import time
from typing import Any

CALL_LOG: list[dict[str, Any]] = []
"""
Process-wide record of every logged tool/model call, for main.py's demo
scripts to print a summary at the end of a run. This is plain in-memory
observability, not pipeline data -- deliberately kept out of ADK session
state so it doesn't show up in the 'FINAL SESSION STATE' dump alongside the
actual data quality / valuation results.
"""

_tool_call_starts: dict[str, float] = {}
_model_call_starts: dict[int, float] = {}


def reset_call_log() -> None:
    CALL_LOG.clear()
    _tool_call_starts.clear()
    _model_call_starts.clear()


def summarize_call_log() -> str:
    if not CALL_LOG:
        return "(no calls logged)"
    lines = []
    for entry in CALL_LOG:
        if entry["kind"] == "tool":
            status = "BLOCKED" if entry.get("blocked") else "ok"
            lines.append(f"  [tool]  {entry['name']:<28} latency_ms={entry['latency_ms']!s:<8} {status}")
        else:
            lines.append(
                f"  [model] {entry['agent']:<28} latency_ms={entry['latency_ms']!s:<8} "
                f"prompt_tokens={entry.get('prompt_tokens')} output_tokens={entry.get('output_tokens')}"
            )
    return "\n".join(lines)


def _tool_call_key(tool, tool_context) -> str:
    # function_call_id uniquely identifies one tool invocation, even when
    # several calls to the same tool happen concurrently (e.g. across
    # ParallelAgent branches) or ADK hands before/after different Context
    # instances for the same call. Falls back to object identity if a given
    # tool/context implementation doesn't set it.
    call_id = getattr(tool_context, "function_call_id", None)
    return f"{tool.name}:{call_id or id(tool_context)}"


def before_tool_log(tool, args: dict, tool_context) -> None:
    _tool_call_starts[_tool_call_key(tool, tool_context)] = time.monotonic()
    return None


def after_tool_log(tool, args: dict, tool_context, tool_response: dict) -> None:
    start = _tool_call_starts.pop(_tool_call_key(tool, tool_context), None)
    latency_ms = round((time.monotonic() - start) * 1000, 1) if start is not None else None
    CALL_LOG.append({"kind": "tool", "name": tool.name, "args": args, "latency_ms": latency_ms})
    print(f"[callback] tool={tool.name} latency_ms={latency_ms} args={args}")
    return None


def before_model_log(callback_context, llm_request) -> None:
    # ADK invokes before/after_model_callback with keyword args named
    # exactly `callback_context`/`llm_request`/`llm_response` -- the
    # parameter names below aren't cosmetic, they have to match those
    # keywords or the call raises TypeError: unexpected keyword argument.
    _model_call_starts[id(callback_context)] = time.monotonic()
    return None


def after_model_log(callback_context, llm_response) -> None:
    start = _model_call_starts.pop(id(callback_context), None)
    latency_ms = round((time.monotonic() - start) * 1000, 1) if start is not None else None

    usage = getattr(llm_response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    agent_name = getattr(callback_context, "agent_name", "?")
    CALL_LOG.append(
        {
            "kind": "model",
            "agent": agent_name,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
    )
    print(
        f"[callback] model_call agent={agent_name} latency_ms={latency_ms} "
        f"prompt_tokens={prompt_tokens} output_tokens={output_tokens}"
    )
    return None


def validate_valuation_input(tool, args: dict, tool_context) -> dict | None:
    """
    Guardrail: before_tool_callback on run_valuation_model.

    run_valuation_model_fn (real_estate_agents/tools.py) does `record["sqft"] - 1800`
    with no validation of its own -- a missing/non-numeric sqft crashes the
    tool outright. That shouldn't happen if the data quality agent did its
    job, but the valuation explainer agent decides *what to pass* to this
    tool call itself (it's instructed to call it on 'final_record', not
    handed the dict directly) -- an LLM can still restate that dict wrong.
    A callback catches that at the tool boundary, independent of whether
    the calling agent's prompt was followed correctly.

    Returning a dict here short-circuits the real tool call: ADK uses it as
    the tool's response instead of invoking run_valuation_model_fn.
    """
    if tool.name != "run_valuation_model":
        return None

    record = args.get("record")
    sqft = record.get("sqft") if isinstance(record, dict) else None

    if not isinstance(sqft, (int, float)) or isinstance(sqft, bool) or sqft <= 0:
        print(f"[guardrail] blocked run_valuation_model: invalid sqft={sqft!r} in record={record!r}")
        CALL_LOG.append({"kind": "tool", "name": tool.name, "args": args, "latency_ms": 0.0, "blocked": True})
        return {
            "error": "invalid_input",
            "reason": (
                f"record.sqft must be a positive number, got {sqft!r}. Re-check "
                "'final_record' from the data quality decision before calling this tool again."
            ),
        }

    return None
