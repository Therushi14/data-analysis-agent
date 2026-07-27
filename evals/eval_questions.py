"""
Eval harness for the data-analysis agent.

Drop this into evals/ in your project. Each entry has:
  - id          : stable identifier
  - question    : the natural-language prompt you send to your agent
  - difficulty  : easy | medium | hard
  - expected    : the VERIFIED ground-truth answer (computed from sales_data.csv)
  - kind        : "numeric" | "categorical" | "dict"  (how to compare)
  - tests       : what capability this question probes

The runner (`run_evals.py`) selects from EVAL_SET, sends each question to the
agent, and scores the answer with `checkers.py`. This module stays pure data.
"""

from __future__ import annotations

from pathlib import Path

DATA_PATH = Path(__file__).parent / "sales_data.csv"

EVAL_SET = [
    # ---------- easy ----------
    {"id": "q1", "difficulty": "easy", "kind": "numeric",
     "question": "How many orders are in the dataset?",
     "expected": 3000, "tests": "basic loop / row count"},

    {"id": "q2", "difficulty": "easy", "kind": "numeric",
     "question": "What is the total net revenue after discounts?",
     "expected": 28448095.67, "tests": "computed column (revenue) + sum"},

    {"id": "q3", "difficulty": "easy", "kind": "numeric",
     "question": "What is the total profit across all orders?",
     "expected": 12270649.44, "tests": "profit = revenue - cost, then sum"},

    {"id": "q8", "difficulty": "easy", "kind": "numeric",
     "question": "What is the average order value (net revenue per order)?",
     "expected": 9482.70, "tests": "mean of a computed column"},

    {"id": "q15", "difficulty": "easy", "kind": "numeric",
     "question": "How many orders have a missing customer segment?",
     "expected": 90, "tests": "missing-value handling"},

    {"id": "q17", "difficulty": "easy", "kind": "numeric",
     "question": "What is the average discount given, as a percentage?",
     "expected": 2.15, "tests": "mean * 100 formatting"},

    {"id": "q18", "difficulty": "easy", "kind": "numeric",
     "question": "How many distinct products were sold?",
     "expected": 18, "tests": "nunique"},

    {"id": "q19", "difficulty": "easy", "kind": "numeric",
     "question": "What was the highest profit from a single order?",
     "expected": 30099.51, "tests": "max of computed column"},

    # ---------- medium ----------
    {"id": "q4", "difficulty": "medium", "kind": "dict",
     "question": "What is the total revenue by region?",
     "expected": {"North": 8006624.18, "South": 5649149.39,
                  "East": 4799093.36, "West": 9993228.74},
     "tests": "MUST normalize messy region casing/whitespace, then groupby"},

    {"id": "q5", "difficulty": "medium", "kind": "categorical",
     "question": "Which region has the highest revenue?",
     "expected": "West", "tests": "groupby + idxmax after cleaning"},

    {"id": "q6", "difficulty": "medium", "kind": "categorical",
     "question": "Which product category is the most profitable?",
     "expected": "Electronics", "tests": "groupby profit + idxmax"},

    {"id": "q7", "difficulty": "medium", "kind": "dict",
     "question": "What are the top 3 products by total revenue?",
     "expected": {"Docking Station": 3439392.79, "Laptop Pro": 3356761.60,
                  "Monitor 27\"": 3348195.56},
     "tests": "groupby + sort + head(3)"},

    {"id": "q9", "difficulty": "medium", "kind": "categorical",
     "question": "Which sales rep generated the most profit?",
     "expected": "David Okafor", "tests": "groupby rep + idxmax"},

    {"id": "q10", "difficulty": "medium", "kind": "numeric",
     "question": "What is the overall profit margin (profit divided by revenue) as a percentage?",
     "expected": 43.13, "tests": "ratio of two sums * 100"},

    {"id": "q14", "difficulty": "medium", "kind": "numeric",
     "question": "What was the total revenue from the Enterprise segment in Q3?",
     "expected": 2668825.10, "tests": "multi-condition filter + aggregate"},

    {"id": "q16", "difficulty": "medium", "kind": "numeric",
     "question": "What percentage of total revenue came from the Software category?",
     "expected": 13.78, "tests": "filtered sum / total sum"},

    {"id": "q20", "difficulty": "medium", "kind": "dict",
     "question": "What is the profit margin by category, as a percentage?",
     "expected": {"Electronics": 33.42, "Furniture": 43.88,
                  "Office Supplies": 54.06, "Software": 79.51},
     "tests": "grouped ratio of two aggregates"},

    # ---------- hard ----------
    {"id": "q11", "difficulty": "hard", "kind": "numeric",
     "question": "Which month had the highest revenue? Give the month number.",
     "expected": 12, "tests": "parse dates, group by month, idxmax"},

    {"id": "q12", "difficulty": "hard", "kind": "numeric",
     "question": "In which month was the biggest month-over-month revenue drop? Give the month number.",
     "expected": 9, "tests": "sort months, diff consecutive, find min"},

    {"id": "q13", "difficulty": "hard", "kind": "categorical",
     "question": "Which region grew fastest in units sold from Q1 to Q4?",
     "expected": "West", "tests": "pivot region x quarter, % change Q1->Q4, rank"},
]


DIFFICULTIES = ("easy", "medium", "hard")


def get_questions(
    ids: list[str] | None = None,
    difficulty: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Select questions from EVAL_SET by id and/or difficulty, preserving order.

    - `ids`: keep only these ids (order follows EVAL_SET).
    - `difficulty`: one of easy/medium/hard.
    - `limit`: cap the count (handy for staying inside the daily quota).
    """
    items = EVAL_SET
    if difficulty:
        items = [q for q in items if q["difficulty"] == difficulty]
    if ids:
        wanted = set(ids)
        items = [q for q in items if q["id"] in wanted]
    if limit is not None:
        items = items[:limit]
    return items


def by_id(qid: str) -> dict | None:
    return next((q for q in EVAL_SET if q["id"] == qid), None)
