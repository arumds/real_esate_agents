# Evaluating these agents

Agent evaluation here happens at three levels, from cheapest/fastest to
most expensive/holistic. All three matter because they catch different
failure modes — a component eval won't catch a bad handoff between agents,
and an end-to-end eval won't tell you *which* agent broke.

## Level 1 — Deterministic unit tests (no LLM call)

Pure-Python logic that has nothing to do with the LLM's judgment: the
grounding checker, field-completeness rules, JSON-parsing fallbacks. These
are ordinary `pytest`-style tests, run on every commit, zero cost, zero
flakiness.

See `eval/test_deterministic.py`. Example assertions:
- `check_field_completeness` flags an out-of-range sqft
- `grounding_guardrail` rejects a narrative with an invented dollar figure
  and accepts one that only cites real SHAP figures
- malformed LLM JSON falls back to `flag_for_review` rather than crashing

Run: `pytest eval/test_deterministic.py -v`

## Level 2 — Golden-dataset agent evals (LLM-in-the-loop, needs API key)

A small labeled dataset of property records with a known *correct*
disposition (`eval/golden_dataset.py`), run through the real agent, scored
against the label. This is the standard way to eval an agent that makes a
classification-shaped decision (pass / auto_correct / flag_for_review):

- **Accuracy / disposition match rate**: does the agent's disposition match
  the golden label?
- **Precision on auto_correct**: of the records the agent chose to
  auto-correct, how many corrections were actually right? (False positives
  here are the costly failure mode — a wrong auto-correction silently
  corrupts data feeding the valuation model.)
- **Recall on flag-worthy records**: of the records that truly needed human
  review, how many did the agent actually flag? (False negatives here are
  the costly failure mode — bad data reaching the model undetected.)
- **Calibration**: do the agent's `confidence` scores actually track its
  accuracy? (e.g. bucket predictions by confidence, check empirical
  accuracy per bucket)

For the Explainable Valuation Agent, "correctness" isn't a single label —
it's a narrative — so the eval is different in kind:
- **Grounding rate**: % of narratives that pass the grounding guardrail on
  the first attempt (no retry needed) — a proxy for how often the base
  model hallucinates numbers before the guardrail catches it.
- **LLM-as-judge rubric scoring**: a separate LLM call scores the narrative
  1-5 on faithfulness, clarity for the target audience, and completeness
  (did it mention the top 2-3 drivers of value?). This is the standard
  approach for evaluating open-ended generation where there's no single
  correct string to match against.

See `eval/run_golden_eval.py` — runs both agents over the golden dataset and
prints a scorecard.

## Level 3 — End-to-end / trajectory eval

Because these are *agents* (multi-step, tool-calling, sometimes
multi-agent under CrewAI's hierarchical process), evaluating only the final
answer misses failures in the reasoning trajectory itself:

- **Tool-call correctness**: did the agent call the *right* tools? (e.g.
  did it actually cross-check the assessor record when a parcel_id was
  present, or did it skip straight to a disposition?) This is checked
  against the `transcript` (hand-rolled version) or CrewAI's task/tool-usage
  logs (CrewAI version).
- **Delegation correctness (CrewAI-specific)**: under `Process.hierarchical`,
  did the manager agent correctly withhold delegation of the valuation task
  when the data quality task returned `flag_for_review`? This is a routing
  behavior that can silently break if a prompt edit changes the manager's
  judgment — it needs its own explicit test case, not just "the final
  narrative looked fine."
- **Cost/latency budget**: number of LLM calls and tool calls per record,
  tracked over time so a prompt change that triples tool calls doesn't
  ship unnoticed.
- **Regression suite**: re-run Level 2's golden dataset after every prompt
  or model change and diff the scorecard — this is the same idea as
  regression testing in normal software engineering, applied to prompts.

## Where human review fits in

None of the above replaces a human-review sampling process in production:
route a random % of `pass` and `auto_correct` dispositions to a human
reviewer on an ongoing basis (not just at launch) to catch drift the golden
dataset doesn't cover — golden datasets go stale as the real data
distribution shifts (new neighborhoods, new listing-feed quirks), and only
live sampling catches that.
