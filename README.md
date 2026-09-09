# Multi-agent Agentic AI Projects

Two runnable agentic AI projects that sit around your existing valuation /
rental prediction ML models — they don't replace the models, they clean the
data going in and explain the output coming out. Alongside them, two
multi-agent orchestration refactors (Google ADK, and a custom Python
pipeline) and a real LangChain RAG pipeline shared between them.

```
real_estate_agents/
├── shared/
│   ├── llm_client.py      # OpenAI API wrapper + generic agent loop (ReAct-style)
│   └── mcp_base.py        # Helper to expose any tool set as an MCP server
├── data_quality_agent/
│   ├── tools.py           # Validators + mocked secondary-source lookups
│   ├── agent.py           # Orchestration: rules first, LLM for judgment calls
│   └── mcp_server.py      # Exposes the agent + tools over MCP
├── explainable_valuation_agent/
│   ├── rag.py             # In-memory vector store over comps/market reports
│   ├── shap_utils.py      # SHAP formatting + narrative grounding checker
│   ├── agent.py           # Retrieve -> draft -> self-critique -> redraft loop
│   └── mcp_server.py      # Exposes the agent + tools over MCP
├── orchestrate_pipeline.py # Chains both agents around a (mocked) ML model call
├── langchain_rag/          # Real LangChain RAG pipeline, consumed by adk_version
│   ├── embeddings.py        # Custom Embeddings adapter reusing shared/llm_client.py::embed()
│   ├── retriever.py         # RecursiveCharacterTextSplitter + FAISS vectorstore
│   └── tool.py               # The single @tool consumed by adk_version
├── adk_version/            # Multi-agent system using Google ADK
│   ├── schemas.py          # Pydantic output schemas for each agent (independent copy)
│   ├── tools.py            # Wraps the same tool functions as FunctionTool for ADK
│   ├── agent.py            # Both LlmAgents + PipelineOrchestratorAgent + root_agent (ADK's auto-discovery entry point)
│   ├── callbacks.py        # before/after_tool + before/after_model logging, plus an input-validation guardrail
│   ├── grounding_checker.py# Custom BaseAgent implementing the output-side guardrail manually
│   └── main.py             # Entry point using Runner + InMemorySessionService
├── eval/                   # Evaluation harness (see eval/README.md)
│   ├── test_deterministic.py
│   ├── golden_dataset.py
│   └── run_golden_eval.py
├── genai_engineer_prep_plan.md # Personal study/prep notes, not part of the pipeline
└── requirements.txt
```

## Is this multi-agent?

The original two agents (`data_quality_agent/`, `explainable_valuation_agent/`)
are separate agents with distinct roles, but they're chained in a **fixed
sequence** by `orchestrate_pipeline.py` — no delegation, no runtime decision
about who does what.

`adk_version/` is the real multi-agent refactor, built on Google's Agent
Development Kit. Same two underlying tasks, same shared `tools.py`/`rag.py`/
`shap_utils.py` logic underneath, but orchestrated through ADK's own
primitives instead of a hand-rolled Python script. Everything agent-related
— both `LlmAgent`s, the orchestrator that wires them together, and
`root_agent` — lives in one `agent.py`, following ADK's own convention:
`adk web`/`adk run`/`adk deploy` auto-discover a module-level `root_agent`
by importing this file.

- **Delegation**: `PipelineOrchestratorAgent` (in `agent.py`) is a custom
  `BaseAgent` that makes the "never value a flagged record" call with a
  plain Python `if` statement rather than trusting an LLM's judgment on a
  rule that must never be violated — deterministic routing for a rule with
  zero tolerance for error.
- **Guardrail/retry**: `grounding_checker.py` reimplements the same
  grounding check from the hand-rolled version as a custom non-LLM
  `BaseAgent`, sitting inside a `LoopAgent` alongside the explainer agent,
  escalating (breaking the loop) only once the check passes.
- **Fan-out**: `agent.py::build_multi_audience_pipeline` uses a
  `ParallelAgent` to generate the homeowner/underwriter/appraiser
  narratives concurrently — one `(explainer, grounding_checker)` `LoopAgent`
  per audience, each writing to its own `valuation_explanation_raw__<audience>`
  session-state key so the concurrent branches don't clobber each other.
  `PipelineOrchestratorAgent` doesn't care whether it's handed this
  `ParallelAgent` or the single-audience `LoopAgent` — same deterministic
  gate either way.
- **Callbacks**: `callbacks.py` attaches `before/after_tool_callback` and
  `before/after_model_callback` hooks to both `LlmAgent`s, logging every
  tool/model call's latency (and token counts, when the model returns
  `usage_metadata`) into an in-memory `CALL_LOG` that `main.py` prints as a
  summary at the end of each run. These always return `None`, so they never
  change pipeline behavior — pure observability. The valuation explainer
  also carries `validate_valuation_input`, a `before_tool_callback`
  guardrail on `run_valuation_model`: it rejects a call whose `record` is
  missing a valid `sqft` *before* the tool runs (which would otherwise
  crash on `record["sqft"] - 1800`), returning an error dict that ADK uses
  as the tool's response instead. This is a different guardrail mechanism
  than `grounding_checker.py`'s: the checker validates the explainer's
  *output* after generation inside a `LoopAgent`; this guardrail validates
  a tool's *input* before it's called, at the callback layer.
- **Model provider**: ADK defaults to Gemini. Using OpenAI (to stay
  consistent with the rest of this repo) requires ADK's LiteLLM bridge:
  `pip install "google-adk[extensions]"`.

Run it:
```bash
pip install "google-adk[extensions]"
export OPENAI_API_KEY=sk-...
python -m adk_version.main   # runs both the single-audience and multi-audience (ParallelAgent) demos
```

Or explore it interactively via ADK's own dev UI, which discovers `root_agent`
in `adk_version/agent.py` automatically:
```bash
adk web   # run from the repo root; open the printed localhost URL, pick "adk_version"
```
Note: `adk web` seeds session state from your chat message, not from a
Python dict — `main.py`'s pre-seeded `input_record`/`audience`/
`property_summary` state doesn't happen for you there, so you'd need to
paste the record into the chat and adjust the agent instructions (or add a
`before_agent_callback`) to parse it, rather than reading it pre-seeded from
session state.

## A real RAG pipeline: LangChain

`langchain_rag/tool.py` is a **real RAG pipeline** — `RecursiveCharacterTextSplitter`
chunks the comp/market-report corpus, a `SharedLLMEmbeddings` adapter (wrapping
this repo's existing `shared/llm_client.py::embed()`, so no second embeddings
provider) feeds a `FAISS` vectorstore, and a LangChain `@tool` wraps the
resulting similarity search. `adk_version/tools.py` consumes this tool
directly via ADK's own LangChain bridge:

```python
# adk_version/tools.py
from google.adk.tools.langchain_tool import LangchainTool
retrieve_market_context = LangchainTool(_lc_retrieve_market_context)
```

The hand-rolled version (`explainable_valuation_agent/rag.py`) deliberately
keeps its original simple in-memory cosine-similarity implementation
untouched, as a baseline for comparison against this real RAG pipeline.

## 1. Data Quality Agent

**Problem:** incoming property records have wrong sqft, mismatched
addresses, missing fields — garbage in, garbage valuations out.

**How it works:**
1. Deterministic rule checks run first (`tools.py::check_field_completeness`)
   — no LLM call needed for structurally sound records with no parcel_id to
   cross-check.
2. If there's a parcel_id (something to verify against) or a rule violation,
   an LLM agent loop takes over: it decides *which* secondary-source tools to
   call (`geocode_and_validate_address`, `lookup_county_assessor_record`,
   `compare_reported_vs_authoritative`), inspects the results, and returns a
   disposition:
   - `pass` — clean, send to the model
   - `auto_correct` — confident fix from an authoritative source, with a
     full audit trail of what changed and why
   - `flag_for_review` — ambiguous, kicked to a human
3. Everything (tool calls, reasoning, final decision) is captured in a
   transcript for audit — important for a regulated valuation pipeline where
   you need to explain why a record was auto-corrected.

Run it:
```bash
python -m data_quality_agent.agent
```

## 2. Explainable Valuation Agent

**Problem:** your model outputs "$312,000" and a SHAP vector — useless to an
underwriter or homeowner without translation.

**How it works:**
1. **RAG**: retrieves relevant comps and market-report snippets for the
   property (`rag.py`) so the narrative can reference real market context,
   not just raw feature numbers.
2. Formats the SHAP contributions into a ranked, compact block
   (`shap_utils.py`).
3. LLM drafts a narrative, tone-matched to the audience (`underwriter` /
   `homeowner` / `appraiser`).
4. **Self-critique loop**: a deterministic grounding checker verifies every
   dollar figure in the draft traces back to the actual SHAP data. If the
   LLM invented or misquoted a number, the agent feeds that back and asks
   for a redraft — up to a small retry budget — instead of shipping an
   ungrounded explanation.

Run it:
```bash
python -m explainable_valuation_agent.agent
```

## End-to-end pipeline

```bash
python orchestrate_pipeline.py
```
Chains: raw record → Data Quality Agent → (mocked) valuation model → SHAP →
Explainable Valuation Agent → final narrative. Swap `mock_valuation_model()`
for your real model + `shap.TreeExplainer(...)` call — nothing else changes.

## Running with a real OpenAI key

Both agents run in `MOCK_MODE` by default (see `shared/llm_client.py`) so
you can execute and read through the full control flow without any API
key or cost. Set a key to see real generations and real agent reasoning:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python orchestrate_pipeline.py
```

Note: in mock mode, the Data Quality Agent's LLM step can't actually reason
over tool results, so it deliberately **fails closed to `flag_for_review`**
rather than guess — with a real key and a record whose sqft matches the
assessor record, it correctly returns `pass`. This fail-closed default is a
reasonable production posture for a regulated pipeline too: uncertainty
should route to a human, not to a guess.

## Exposing both agents via MCP (agent interoperability)

```bash
pip install mcp
python -m data_quality_agent.mcp_server            # stdio MCP server
python -m explainable_valuation_agent.mcp_server    # stdio MCP server
```

Point any MCP-compatible host (Claude Desktop's config, another team's
agent orchestrator, a Claude Agent SDK app) at these servers and they can
call `run_data_quality_check` or `explain_valuation` without any
custom client code — the whole point of standardizing on MCP instead of a
bespoke REST endpoint per agent. `orchestrate_pipeline.py` calls the agents
in-process for simplicity; in a multi-team deployment you'd swap those
direct calls for an MCP client talking to each server over stdio/SSE.

## Concept coverage

| Concept | Where |
|---|---|
| **LLM APIs (OpenAI)** | `shared/llm_client.py` — single integration point, native function/tool calling, chat + embeddings |
| **AI agents** | Both `agent.py` files: the LLM decides which tools to call and when to stop, not a fixed script |
| **Agentic frameworks** | `shared/llm_client.py::run_agent_loop` — a minimal, dependency-free ReAct-style loop (ports directly to LangChain/LlamaIndex/Claude Agent SDK executors if you want a heavier framework later; kept hand-rolled here so the control flow is fully auditable, which matters for a regulated valuation domain); `adk_version/` shows the same problem re-orchestrated with Google ADK, incl. `LoopAgent` (grounding retry) and `ParallelAgent` (concurrent per-audience narratives) |
| **RAG** | `explainable_valuation_agent/rag.py` (hand-rolled) and `langchain_rag/` (real LangChain + FAISS pipeline, consumed by `adk_version/`) |
| **MCP (agent interoperability)** | `shared/mcp_base.py` + both `mcp_server.py` files — tools exposed as standard MCP servers, callable by any compatible host |

## How are these agents evaluated?

Full detail in `eval/README.md`, but the short version — evaluation happens
at three levels:

1. **Deterministic unit tests** (`eval/test_deterministic.py`, no LLM,
   no API key) — checks the pure-Python logic: rule-based validators, the
   grounding guardrail, and safe fallback behavior on malformed LLM output.
   Run: `pytest eval/test_deterministic.py -v`
2. **Golden-dataset accuracy eval** (`eval/golden_dataset.py` +
   `eval/run_golden_eval.py`, needs a real API key for a meaningful score) —
   labeled property records with known-correct dispositions, scored for
   overall accuracy, precision on `auto_correct` (false positives silently
   corrupt data), and recall on records that should be flagged (false
   negatives let bad data reach the model). Run: `python -m eval.run_golden_eval`
3. **End-to-end / trajectory eval** — did the agent call the *right* tools?
   This needs transcript/log inspection, not just the final answer, since a
   multi-step agent can reach the right answer via the wrong reasoning path.

None of this replaces ongoing human-review sampling in production — golden
datasets go stale as real data drifts, live sampling doesn't.

## Extending toward production

- Swap `tools.py`'s mock county-assessor/geocoder lookups for real API calls
  (county open-data portals, USPS/Smarty, a geocoder).
- Swap `rag.py`'s in-memory store for Chroma/pgvector and point it at a real
  ingestion job over your MLS comp feed and market reports.
- Wire `orchestrate_pipeline.py::mock_valuation_model` to your real model +
  `shap.TreeExplainer(model).shap_values(X)`.
- Add persistence for the audit transcripts (`DataQualityResult.transcript`)
  — e.g. write to a database table for compliance review.
- Consider a confidence threshold policy: e.g. `auto_correct` only allowed
  when `confidence >= 0.9`, else always `flag_for_review`.
