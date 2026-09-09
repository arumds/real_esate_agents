# Multi-agent Agentic AI Projects

Two agents that sit around your existing valuation / rental prediction ML
models — they don't replace the models, they clean the data going in and
explain the output coming out. One agent implementation, built on Google's
Agent Development Kit (`real_estate_agents/`), consumed three different ways:
directly (`real_estate_agents/main.py` or `adk web`), over MCP
(`data_quality_agent/mcp_server.py`, `explainable_valuation_agent/mcp_server.py`),
and by the eval harness (`eval/run_golden_eval.py`). A real LangChain RAG
pipeline (`langchain_rag/`) backs the valuation explainer. `data_quality_agent/`
and `explainable_valuation_agent/` hold deterministic tools/utilities and an
MCP transport layer; all agent reasoning lives in `real_estate_agents/agent.py`.

```
real_estate_agents/
├── shared/
│   ├── llm_client.py      # OpenAI wrapper: chat/embed. embed() feeds rag.py and langchain_rag/embeddings.py
│   └── mcp_base.py        # Helper to expose any tool set as an MCP server
├── data_quality_agent/
│   ├── tools.py           # Validators + mocked secondary-source lookups (single source
│   │                       # of truth -- also wrapped as FunctionTools in real_estate_agents/tools.py)
│   └── mcp_server.py      # MCP transport: runs the data_quality_agent from real_estate_agents/ via runner_utils.py
├── explainable_valuation_agent/
│   ├── rag.py             # Simple in-memory vector store -- a baseline to compare
│   │                       # against langchain_rag/'s real pipeline
│   ├── shap_utils.py      # SHAP formatting + a grounding-check reference implementation
│   └── mcp_server.py      # MCP transport: runs the valuation loop from real_estate_agents/ via runner_utils.py
├── langchain_rag/          # Real LangChain RAG pipeline, consumed by real_estate_agents/
│   ├── embeddings.py        # Custom Embeddings adapter reusing shared/llm_client.py::embed()
│   ├── retriever.py         # RecursiveCharacterTextSplitter + FAISS vectorstore
│   └── tool.py               # The single @tool consumed by real_estate_agents/
├── real_estate_agents/     # The agent implementation -- everything else calls into this
│   ├── schemas.py          # Pydantic output schemas (DataQualityDecision, ValuationExplanation)
│   ├── tools.py            # Wraps data_quality_agent/tools.py + langchain_rag as ADK tools
│   ├── agent.py            # Both LlmAgents + PipelineOrchestratorAgent + root_agent (ADK's auto-discovery entry point)
│   ├── callbacks.py        # before/after_tool + before/after_model logging, plus an input-validation guardrail
│   ├── grounding_checker.py# Custom BaseAgent implementing the output-side guardrail
│   ├── runner_utils.py     # run_agent(agent, state, trigger_text) -- lets sync callers
│   │                       # (the MCP servers, eval/run_golden_eval.py) call an ADK agent
│   │                       # like a plain function instead of each wiring Runner/SessionService
│   └── main.py             # Demo entry point using Runner + InMemorySessionService
├── eval/                   # Evaluation harness (see eval/README.md)
│   ├── test_deterministic.py  # Pure-Python logic only -- no LLM, no ADK
│   ├── golden_dataset.py
│   └── run_golden_eval.py     # Runs the data_quality_agent from real_estate_agents/ over the golden dataset
├── genai_engineer_prep_plan.md # Personal study/prep notes, not part of the pipeline
└── requirements.txt
```

## The agent implementation: Google ADK

`real_estate_agents/agent.py` holds everything agent-related: both `LlmAgent`s, the
orchestrator that wires them together, and `root_agent` — the module-level
variable ADK's own tooling (`adk web`/`adk run`/`adk deploy`) auto-discovers
by importing this file.

- **Delegation**: `PipelineOrchestratorAgent` is a custom `BaseAgent` that
  makes the "never value a flagged record" call with a plain Python `if`
  statement rather than an LLM's judgment — deterministic routing for a
  rule with zero tolerance for error.
- **Guardrail/retry**: `grounding_checker.py` implements a grounding check
  as a custom non-LLM `BaseAgent` sitting inside a `LoopAgent` alongside the
  explainer agent, escalating (breaking the loop) only once the check
  passes.
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
  as the tool's response instead. This is a different mechanism than
  `grounding_checker.py`'s: the checker validates the explainer's *output*
  after generation inside a `LoopAgent`; this guardrail validates a tool's
  *input* before it's called, at the callback layer.
- **Model provider**: ADK defaults to Gemini. Using OpenAI (to stay
  consistent with the rest of this repo) requires ADK's LiteLLM bridge:
  `pip install "google-adk[extensions]"`. There's no offline/mock mode —
  every run calls a real model.

Run it end to end:
```bash
pip install "google-adk[extensions]"
export OPENAI_API_KEY=sk-...   # or: export ADK_MODEL=gemini-2.0-flash GOOGLE_API_KEY=...
python -m real_estate_agents.main     # runs both the single-audience and multi-audience (ParallelAgent) demos
```

Or explore it interactively via ADK's own CLI/dev UI, which discover
`root_agent` in `real_estate_agents/agent.py` automatically (both auto-discover
by directory name, so run them from the repo root):
```bash
adk run real_estate_agents   # terminal chat with root_agent
adk web                      # browser UI at the printed localhost URL; pick "real_estate_agents"
```
Unlike `main.py` (which pre-seeds session state with `input_record`/
`audience`/`property_summary` before the run starts), `adk run`/`adk web`
only pass along whatever you type as a chat message — `DATA_QUALITY_INSTRUCTION`
in `agent.py` handles that by falling back to reading the record straight
out of the message when session state doesn't have it yet.

## How the two domain packages fit in

`data_quality_agent/` and `explainable_valuation_agent/` hold the
deterministic tool functions each ADK `FunctionTool` wraps
(`data_quality_agent/tools.py`), a simple RAG baseline
(`explainable_valuation_agent/rag.py`), and an `mcp_server.py` per package
that exposes an ADK agent over MCP:

- `data_quality_agent/mcp_server.py::run_data_quality_check` builds
  `real_estate_agents.agent.build_data_quality_agent()` and runs it via
  `real_estate_agents/runner_utils.py::run_agent()`.
- `explainable_valuation_agent/mcp_server.py::explain_valuation` builds the
  **same `(valuation_explainer_agent, grounding_checker)` `LoopAgent`**
  `real_estate_agents/agent.py::build_pipeline()` uses for this step — not the
  bare `LlmAgent` — so a narrative that fails the grounding check gets
  retried over MCP too, not just when run through `real_estate_agents/main.py`.

**API shape worth knowing**: `explain_valuation` takes a property `record`
(not precomputed `predicted_price`/`base_value`/`contributions`), because
the ADK agent calls `run_valuation_model` itself as part of its own tool
loop — the tool runs both the (mocked) model call and the narrative step
together. A caller who already has externally computed SHAP contributions
and only wants narration would need a different tool shape than this one.

`real_estate_agents/runner_utils.py::run_agent(agent, state, trigger_text)` is
what makes this possible without each caller re-deriving ADK's
Runner/SessionService plumbing: it creates a fresh in-memory session,
seeds it with `state`, drives the agent to completion via `Runner.run_async`,
and returns the final session state as a plain dict — the boundary between
ADK's async world and the rest of this repo, which is otherwise sync
throughout (FastMCP tool functions and `eval/run_golden_eval.py`'s loop are
both plain sync callables).

Run the MCP servers:
```bash
pip install mcp "google-adk[extensions]"
export OPENAI_API_KEY=sk-...   # or ADK_MODEL + GOOGLE_API_KEY for Gemini
python -m data_quality_agent.mcp_server            # stdio MCP server
python -m explainable_valuation_agent.mcp_server    # stdio MCP server
```
Point any MCP-compatible host (Claude Desktop's config, another team's
agent orchestrator, a Claude Agent SDK app) at these servers and they can
call `run_data_quality_check` or `explain_valuation` without any custom
client code.

## A real RAG pipeline: LangChain

`langchain_rag/tool.py` is a **real RAG pipeline** — `RecursiveCharacterTextSplitter`
chunks the comp/market-report corpus, a `SharedLLMEmbeddings` adapter (wrapping
this repo's existing `shared/llm_client.py::embed()`, so no second embeddings
provider) feeds a `FAISS` vectorstore, and a LangChain `@tool` wraps the
resulting similarity search. `real_estate_agents/tools.py` consumes this tool
directly via ADK's own LangChain bridge:

```python
# real_estate_agents/tools.py
from google.adk.tools.langchain_tool import LangchainTool
retrieve_market_context = LangchainTool(_lc_retrieve_market_context)
```

`explainable_valuation_agent/rag.py` is a simple in-memory cosine-similarity
implementation, kept as a baseline for comparison against this real RAG
pipeline. It's exposed as its own standalone `retrieve_market_context` MCP
tool, independent of the ADK agent's internal (LangChain-backed) retrieval.

## 1. Data Quality Agent

**Problem:** incoming property records have wrong sqft, mismatched
addresses, missing fields — garbage in, garbage valuations out.

**How it works** (`real_estate_agents/agent.py::build_data_quality_agent`):
1. The agent's instruction always cross-checks a suspicious/present
   `parcel_id` against the county assessor record via its tools
   (`geocode_and_validate_address`, `lookup_county_assessor_record`,
   `compare_reported_vs_authoritative` — all from `data_quality_agent/tools.py`).
2. It decides a final disposition, enforced via `output_schema=DataQualityDecision`:
   - `pass` — clean, send to the model
   - `auto_correct` — confident fix from an authoritative source, with a
     full audit trail of what changed and why
   - `flag_for_review` — ambiguous, kicked to a human
3. `before/after_tool_callback` (`real_estate_agents/callbacks.py`) logs every tool
   call's latency for observability.

Run it: `python -m real_estate_agents.main`, or via MCP: `python -m data_quality_agent.mcp_server`.

## 2. Explainable Valuation Agent

**Problem:** your model outputs "$312,000" and a SHAP vector — useless to an
underwriter or homeowner without translation.

**How it works** (`real_estate_agents/agent.py::build_valuation_explainer_agent`,
run inside a `LoopAgent` with `grounding_checker.py::GroundingCheckerAgent`):
1. Calls `run_valuation_model` (the mocked ML model + SHAP stand-in) on the
   cleaned record, then `retrieve_market_context` (the real LangChain/FAISS
   RAG pipeline) for market color.
2. Drafts a narrative, tone-matched to the audience (`underwriter` /
   `homeowner` / `appraiser`).
3. **Self-critique loop**: `GroundingCheckerAgent` deterministically verifies
   every dollar figure in the draft traces back to the actual model output.
   If not, it writes feedback into session state and the `LoopAgent` runs
   the explainer again (up to `MAX_GROUNDING_ATTEMPTS`) instead of shipping
   an ungrounded explanation.
4. A `before_tool_callback` guardrail (`validate_valuation_input`) rejects a
   call to `run_valuation_model` whose record is missing a valid `sqft`
   before the tool can crash on it.

Run it: `python -m real_estate_agents.main`, or via MCP: `python -m explainable_valuation_agent.mcp_server`.

## Running with a real API key

ADK's `LlmAgent` always calls a real model — no offline/mock mode. Two
options:
```bash
# OpenAI, via ADK's LiteLLM bridge (default)
pip install "google-adk[extensions]"
export OPENAI_API_KEY=sk-...

# or Gemini natively
export ADK_MODEL=gemini-2.0-flash
export GOOGLE_API_KEY=...
```
Then `python -m real_estate_agents.main`, `python -m data_quality_agent.mcp_server`,
`python -m explainable_valuation_agent.mcp_server`, or `python -m eval.run_golden_eval`
all work the same way — same agents, different entry points.

## Concept coverage

| Concept | Where |
|---|---|
| **LLM APIs** | `real_estate_agents/agent.py::_default_model` — OpenAI via ADK's LiteLLM bridge, or Gemini natively via `ADK_MODEL`/`GOOGLE_API_KEY` |
| **AI agents** | `real_estate_agents/agent.py`'s two `LlmAgent`s: the model decides which tools to call and when to stop, enforced into a structured `output_schema` on the way out |
| **Agentic frameworks** | Google ADK — `LoopAgent` (grounding retry), `ParallelAgent` (concurrent per-audience narratives), and a custom `BaseAgent` (`PipelineOrchestratorAgent`) for the one routing decision that must stay deterministic |
| **RAG** | `explainable_valuation_agent/rag.py` (hand-rolled baseline) and `langchain_rag/` (real LangChain + FAISS pipeline, what the ADK agent actually uses) |
| **MCP (agent interoperability)** | `shared/mcp_base.py` + both `mcp_server.py` files — each runs an ADK agent via `real_estate_agents/runner_utils.py` and exposes it as a standard MCP tool, callable by any compatible host |

## How are these agents evaluated?

Full detail in `eval/README.md`, but the short version — evaluation happens
at two levels: ADK's `output_schema` enforces valid, schema-conformant JSON
at the model API level, so there's no need for a defensive-parsing test on
malformed LLM output.

1. **Deterministic unit tests** (`eval/test_deterministic.py`, no LLM, no
   API key) — checks pure-Python logic that has nothing to do with agent
   implementation: rule-based validators (`data_quality_agent/tools.py`)
   and the grounding-check algorithm (`explainable_valuation_agent/shap_utils.py`).
   Run: `pytest eval/test_deterministic.py -v`
2. **Golden-dataset accuracy eval** (`eval/golden_dataset.py` +
   `eval/run_golden_eval.py`) — labeled property records with
   known-correct dispositions, run through `real_estate_agents/`'s real
   `data_quality_agent` (one fresh agent + session per case, so this calls
   a real model and costs real API calls), scored for overall accuracy,
   precision on `auto_correct`, and recall on records that should be
   flagged. Run: `python -m eval.run_golden_eval`

None of this replaces ongoing human-review sampling in production — golden
datasets go stale as real data drifts, live sampling doesn't.

## Extending toward production

- Swap `data_quality_agent/tools.py`'s mock county-assessor/geocoder
  lookups for real API calls (county open-data portals, USPS/Smarty, a
  geocoder).
- Swap `langchain_rag/retriever.py`'s FAISS store for Chroma/pgvector and
  point it at a real ingestion job over your MLS comp feed and market
  reports.
- Wire `real_estate_agents/tools.py::run_valuation_model_fn` to your real model +
  `shap.TreeExplainer(model).shap_values(X)`.
- Add persistence for the audit trail (today it's just `CALL_LOG` in
  `real_estate_agents/callbacks.py`, in-memory and process-lifetime only) — e.g.
  write tool-call/decision history to a database table for compliance
  review.
- Consider a confidence threshold policy: e.g. `auto_correct` only allowed
  when `confidence >= 0.9`, else always `flag_for_review`.
