"""
shared/llm_client.py

Thin wrapper around the OpenAI Chat Completions API that adds:
  - native function/tool calling (the mechanism every agent in this repo uses
    to let the LLM decide *when* to call a tool, vs. hardcoded control flow)
  - a MOCK_MODE fallback so the two agent projects in this repo run and can be
    demoed end-to-end even without an OPENAI_API_KEY set. This is purely for
    local development/demo purposes -- swap MOCK_MODE off in production.

Concept coverage:
  - "LLM APIs (OpenAI)": this file is the single integration point with the
    OpenAI API. Every agent goes through here, so swapping models/providers
    (e.g. Azure OpenAI, Anthropic) only requires changing this one file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MOCK_MODE = os.getenv("OPENAI_API_KEY") is None

if not MOCK_MODE:
    from openai import OpenAI

    _client = OpenAI()
else:
    _client = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


def _mock_response(system_prompt: str, messages: list[dict], tools: Optional[list[dict]]) -> LLMResponse:
    """
    Deterministic offline stand-in for the OpenAI API so the agents in this
    repo are runnable/demo-able without network access or an API key.

    Real deployments should delete this branch (or keep it behind an explicit
    `--mock` flag) once a key is configured.
    """
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # If tools are offered and this looks like the first turn of a
    # validation/explanation task, "decide" to call the first tool offered.
    # This mimics an LLM's tool-selection behavior deterministically.
    if tools and "TOOL_RESULTS" not in last_user:
        first_tool = tools[0]["function"]
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="mock_call_1", name=first_tool["name"], arguments={})
            ],
        )

    return LLMResponse(
        content=(
            "[MOCK LLM OUTPUT] No OPENAI_API_KEY is set, so this is a canned "
            "response standing in for the model's final answer. Set "
            "OPENAI_API_KEY to see real generations.\n\n"
            f"(Would have reasoned over: {last_user[:200]}...)"
        )
    )


def chat(
    system_prompt: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> LLMResponse:
    """
    Single call to the LLM with optional function/tool definitions.

    `tools` follows the OpenAI tools schema:
        [{"type": "function", "function": {"name": ..., "description": ...,
                                            "parameters": {...JSON schema...}}}]
    """
    if MOCK_MODE:
        return _mock_response(system_prompt, messages, tools)

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=full_messages,
        temperature=temperature,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = _client.chat.completions.create(**kwargs)
    choice = resp.choices[0].message

    tool_calls = []
    if choice.tool_calls:
        for tc in choice.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )

    return LLMResponse(content=choice.content, tool_calls=tool_calls, raw=resp)


def embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Text embeddings used by the RAG layer in the Explainable Valuation Agent.
    Falls back to a cheap deterministic hashing embedding in mock mode so
    cosine similarity still behaves sanely for the demo corpus.
    """
    if MOCK_MODE:
        import hashlib

        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            # 32 bytes -> 32-dim pseudo-embedding, normalized-ish via /255
            vectors.append([b / 255.0 for b in h])
        return vectors

    resp = _client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def run_agent_loop(
    system_prompt: str,
    user_message: str,
    tool_registry: dict[str, Callable[..., Any]],
    tool_specs: list[dict],
    max_turns: int = 6,
) -> tuple[str, list[dict]]:
    """
    Generic ReAct-style agent loop: LLM decides which tool to call (if any),
    we execute it, feed the result back, repeat until the LLM produces a
    final text answer or max_turns is hit.

    This is the "agentic framework" primitive both agents in this repo are
    built on -- a minimal, dependency-free alternative to LangChain/LlamaIndex
    agent executors, so the control flow is fully visible/auditable, which
    matters for a regulated domain like valuations.

    Returns (final_answer, transcript) where transcript is a list of
    {"role", "content"/"tool_calls"/"tool_result"} dicts for audit logging.
    """
    messages: list[dict] = [{"role": "user", "content": user_message}]
    transcript: list[dict] = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = chat(system_prompt, messages, tools=tool_specs)

        if response.tool_calls:
            # Record the assistant's decision to call tool(s)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            transcript.append(
                {"role": "assistant_tool_calls", "tool_calls": [(tc.name, tc.arguments) for tc in response.tool_calls]}
            )

            for tc in response.tool_calls:
                fn = tool_registry.get(tc.name)
                if fn is None:
                    result = {"error": f"Unknown tool '{tc.name}'"}
                else:
                    try:
                        result = fn(**tc.arguments)
                    except Exception as e:  # tool errors are fed back to the LLM, not raised
                        result = {"error": str(e)}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )
                transcript.append({"role": "tool_result", "tool": tc.name, "result": result})

            # In mock mode, mark that tool results have been supplied so the
            # next call to _mock_response produces a final answer instead of
            # calling a tool again.
            if MOCK_MODE:
                messages.append({"role": "user", "content": "TOOL_RESULTS supplied above. Give final answer."})

            continue

        # No tool calls -> final answer
        transcript.append({"role": "assistant_final", "content": response.content})
        return response.content or "", transcript

    return "[Agent stopped: max turns reached without a final answer]", transcript
