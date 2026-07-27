"""Tests for the eval harness: checkers, question selection, and a mock run.

All offline — the end-to-end test uses a fake agent and a temp cache, so it
spends no Gemini quota and touches no real cache/report files.
"""

from __future__ import annotations

from evals import run_evals
from evals.checkers import (
    check_categorical,
    check_dict,
    check_numeric,
    extract_numbers,
)
from evals.eval_questions import EVAL_SET, get_questions


# --- Number extraction ----------------------------------------------------------

def test_extract_handles_currency_commas_and_percent():
    nums = extract_numbers("Total net revenue is $28,448,095.67 at a 43.13% margin.")
    assert 28448095.67 in nums
    assert 43.13 in nums


def test_extract_maps_month_names():
    assert 12.0 in extract_numbers("Revenue peaked in December.")
    assert 9.0 in extract_numbers("The biggest drop was in September.")


# --- Numeric checker ------------------------------------------------------------

def test_numeric_pass_within_tolerance():
    assert check_numeric("about 3000 orders", 3000, tol_pct=0.5).verdict == "pass"


def test_numeric_near_miss_flags_but_fails():
    # gross-vs-net: ~2% off is a near-miss, not a pass, at 0.5% tolerance.
    r = check_numeric("total revenue 29,000,000", 28448095.67, tol_pct=0.5)
    assert r.ok is False
    assert r.verdict == "near"


def test_numeric_fail_when_far():
    assert check_numeric("the answer is 5", 3000, tol_pct=0.5).verdict == "fail"


def test_numeric_fail_when_no_number():
    assert check_numeric("no idea", 3000, tol_pct=0.5).ok is False


# --- Categorical checker --------------------------------------------------------

def test_categorical_word_match():
    assert check_categorical("The top region is West.", "West").ok is True


def test_categorical_no_partial_match():
    # "East" must not match inside "Eastern"
    assert check_categorical("Sales were strong in the Eastern zone.", "East").ok is False


def test_categorical_multiword_name():
    assert check_categorical("David Okafor generated the most profit.", "David Okafor").ok is True


# --- Dict checker ---------------------------------------------------------------

def test_dict_pass_when_all_values_present():
    text = "North: 8006624.18, South: 5649149.39, East: 4799093.36, West: 9993228.74"
    expected = {"North": 8006624.18, "South": 5649149.39,
                "East": 4799093.36, "West": 9993228.74}
    r = check_dict(text, expected, tol_pct=0.5)
    assert r.ok is True and r.matched == 4


def test_dict_partial_is_near_not_pass():
    text = "North: 8006624.18 and West: 9993228.74"
    expected = {"North": 8006624.18, "South": 5649149.39,
                "East": 4799093.36, "West": 9993228.74}
    r = check_dict(text, expected, tol_pct=0.5)
    assert r.ok is False and r.matched == 2


# --- Selection ------------------------------------------------------------------

def test_get_questions_by_difficulty():
    hard = get_questions(difficulty="hard")
    assert {q["id"] for q in hard} == {"q11", "q12", "q13"}


def test_get_questions_by_ids_and_limit():
    assert [q["id"] for q in get_questions(ids=["q2", "q1"])] == ["q1", "q2"]  # EVAL_SET order
    assert len(get_questions(limit=3)) == 3


# --- Mock end-to-end ------------------------------------------------------------

def test_mock_run_scores_full_marks(tmp_path):
    res = run_evals.run(
        agent_fn=run_evals.make_mock_agent_fn(correct=True),
        cache_root=tmp_path,
        out=tmp_path / "report.md",
        verbose=False,
    )
    s = res["summary"]
    assert s["run"] == len(EVAL_SET)        # every question ran
    assert s["passed"] == len(EVAL_SET)     # ground-truth answers all pass
    assert (tmp_path / "report.md").exists()


def test_mock_wrong_agent_fails_numeric(tmp_path):
    res = run_evals.run(
        agent_fn=run_evals.make_mock_agent_fn(correct=False),
        cache_root=tmp_path,
        out=tmp_path / "report.md",
        verbose=False,
    )
    # Wrong numeric answers should not pass; some categorical may still miss too.
    assert res["summary"]["passed"] < len(EVAL_SET)


def test_cache_accumulates_and_report_only_scores_it(tmp_path):
    # First: run just the easy tier into a temp cache.
    run_evals.run(
        difficulty="easy",
        agent_fn=run_evals.make_mock_agent_fn(correct=True),
        cache_root=tmp_path,
        out=tmp_path / "r1.md",
        verbose=False,
    )
    # Then: report-only (no agent) must score the accumulated easy answers.
    res = run_evals.run(
        report_only=True,
        cache_root=tmp_path,
        out=tmp_path / "r2.md",
        verbose=False,
    )
    easy_total = len(get_questions(difficulty="easy"))
    assert res["summary"]["run"] == easy_total
    assert res["summary"]["passed"] == easy_total
