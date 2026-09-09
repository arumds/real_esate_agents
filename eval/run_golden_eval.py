"""
eval/run_golden_eval.py

Runs the Data Quality Agent (hand-rolled version) over the golden dataset
and prints an accuracy scorecard: overall disposition match rate, precision
on auto_correct, and recall on records that should have been flagged.

Works in mock mode (no OPENAI_API_KEY) but the scores will be near-meaningless
then, since the mock LLM can't actually reason -- it's meant to demonstrate
the eval mechanics end-to-end. Set OPENAI_API_KEY for a real accuracy read.

Run:
    python -m eval.run_golden_eval
"""

from __future__ import annotations

from dataclasses import dataclass

from data_quality_agent.agent import run_data_quality_check
from eval.golden_dataset import GOLDEN_DATASET
from shared.llm_client import MOCK_MODE


@dataclass
class CaseResult:
    case_id: str
    expected: str
    actual: str
    correct: bool
    reasoning: str


def run_eval() -> None:
    if MOCK_MODE:
        print(
            "WARNING: OPENAI_API_KEY not set -- running in mock mode. Scores "
            "below reflect the mock LLM's canned behavior, not real agent "
            "accuracy. Set OPENAI_API_KEY for a meaningful read.\n"
        )

    results: list[CaseResult] = []
    for case in GOLDEN_DATASET:
        outcome = run_data_quality_check(case["record"])
        actual = outcome.disposition.value
        results.append(
            CaseResult(
                case_id=case["id"],
                expected=case["expected_disposition"],
                actual=actual,
                correct=(actual == case["expected_disposition"]),
                reasoning=outcome.reasoning,
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
