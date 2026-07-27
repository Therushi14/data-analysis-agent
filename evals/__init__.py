"""Offline eval harness for the data-analysis agent.

Scores the agent against a tiered, verified question set (`eval_questions.py`)
computed from `sales_data.csv`. Built to respect the free-tier quota: results
are cached per question, so a run can be spread across days and the report
scores whatever has accumulated. See `run_evals.py`.
"""
