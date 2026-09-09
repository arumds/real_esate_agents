"""
eval/run_golden_eval.py

Runs the Data Quality Agent (real_estate_agents.agent.build_data_quality_agent)
over the golden dataset and prints an accuracy scorecard: overall
disposition match rate, precision on auto_correct, and recall on records
that should have been flagged.

Needs a real model available -- either OPENAI_API_KEY (via ADK's LiteLLM
bridge, the default) or ADK_MODEL=<gemini model> + GOOGLE_API_KEY. There's
no offline mock mode: ADK's LlmAgent always calls a real model, so this
always reflects real agent accuracy (and costs real API calls -- one fresh
agent run per golden case).

Run:
    python -m eval.run_golden_eval
"""

from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from real_estate_agents.agent import build_data_quality_agent
from real_estate_agents.runner_utils import parse_output, run_agent
from eval.golden_dataset import GOLDEN_DATASET

load_dotenv()


@dataclass
class CaseResult:
    case_id: str
    expected: str
    actual: str
    correct: bool
    reasoning: str


def run_eval() -> None:
    results: list[CaseResult] = []
    for case in GOLDEN_DATASET:
        agent = build_data_quality_agent()
        final_state = run_agent(
            agent,
            state={"input_record": case["record"]},
            trigger_text=f"Validate and explain this property record: {case['record']}",
        )
        decision = parse_output(final_state, "data_quality_decision_raw") or {}
        actual = decision.get("disposition", "flag_for_review")
        results.append(
            CaseResult(
                case_id=case["id"],
                expected=case["expected_disposition"],
                actual=actual,
                correct=(actual == case["expected_disposition"]),
                reasoning=decision.get("reasoning", "Agent produced no decision."),
            )
        )

    # --- Scorecard ---
    total = len(results)
    correct = sum(r.correct for r in results)
    accuracy = correct / total if total else 0.0

    # Recall on records that SHOULD have been flagged: of all golden cases
    # labeled flag_for_review, how many did the agent actually flag?
    should_flag = [r for r in results if r.expected == "flag_for_review"]
    flagged_correctly = [r for r in should_flag if r.actual == "flag_for_review"]
    recall_on_flags = len(flagged_correctly) / len(should_flag) if should_flag else None

    # Precision on auto_correct: of everything the agent chose to
    # auto-correct, how many were labeled auto_correct in the golden set?
    agent_auto_corrected = [r for r in results if r.actual == "auto_correct"]
    correctly_auto_corrected = [r for r in agent_auto_corrected if r.expected == "auto_correct"]
    precision_on_auto_correct = (
        len(correctly_auto_corrected) / len(agent_auto_corrected) if agent_auto_corrected else None
    )

    print("=" * 70)
    print("GOLDEN DATASET SCORECARD -- Data Quality Agent")
    print("=" * 70)
    for r in results:
        mark = "✓" if r.correct else "✗"
        print(f"[{mark}] {r.case_id:30s} expected={r.expected:16s} actual={r.actual}")
        if not r.correct:
            print(f"      reasoning: {r.reasoning[:120]}")

    print("\n--- Metrics ---")
    print(f"Overall disposition accuracy: {correct}/{total} ({accuracy:.0%})")
    print(
        f"Recall on flag-worthy records: "
        f"{len(flagged_correctly)}/{len(should_flag)} "
        f"({recall_on_flags:.0%})" if recall_on_flags is not None else "N/A (no flag cases)"
    )
    print(
        f"Precision on auto_correct: "
        f"{len(correctly_auto_corrected)}/{len(agent_auto_corrected)} "
        f"({precision_on_auto_correct:.0%})" if precision_on_auto_correct is not None else "N/A (agent never auto-corrected)"
    )
    print(
        "\nNote: recall on flag-worthy records matters more than raw accuracy "
        "here -- a false negative (bad data silently passed to the model) is "
        "costlier than a false positive (a clean record sent to human review)."
    )


if __name__ == "__main__":
    run_eval()
