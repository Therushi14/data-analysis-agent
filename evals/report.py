"""Render eval results as a console summary and a markdown scorecard.

A "row" is one dict per EVAL_SET question:
    {id, difficulty, question, kind, expected, verdict, detail,
     answer, status, steps, errors, recovered, tokens, elapsed_s}
`verdict` is "pass" | "near" | "fail" | "pending" (not yet run).
"""

from __future__ import annotations

from datetime import datetime, timezone

_SYMBOL = {"pass": "✅", "near": "🟡", "fail": "❌", "pending": "⬜"}
_DIFFS = ("easy", "medium", "hard")


def summarize(rows: list[dict]) -> dict:
    run = [r for r in rows if r["verdict"] != "pending"]
    passed = [r for r in run if r["verdict"] == "pass"]
    by_diff = {}
    for d in _DIFFS:
        drun = [r for r in run if r["difficulty"] == d]
        dpass = [r for r in drun if r["verdict"] == "pass"]
        dtotal = sum(1 for r in rows if r["difficulty"] == d)
        by_diff[d] = {"passed": len(dpass), "run": len(drun), "total": dtotal}
    tokens = sum(r.get("tokens") or 0 for r in run)
    recov = sum(1 for r in run if r.get("recovered"))
    elapsed = [r.get("elapsed_s") for r in run if r.get("elapsed_s")]
    return {
        "total": len(rows),
        "run": len(run),
        "passed": len(passed),
        "near": sum(1 for r in run if r["verdict"] == "near"),
        "pct": (100.0 * len(passed) / len(run)) if run else 0.0,
        "by_diff": by_diff,
        "tokens": tokens,
        "recovered": recov,
        "avg_elapsed": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
    }


def console_summary(rows: list[dict], meta: dict) -> str:
    s = summarize(rows)
    lines = [""]
    for r in rows:
        sym = _SYMBOL[r["verdict"]]
        lines.append(f"  {sym} {r['id']:<4} [{r['difficulty']:<6}] {r['detail']}")
    lines.append("")
    lines.append(
        f"Score: {s['passed']}/{s['run']} run "
        f"({s['pct']:.0f}%) · coverage {s['run']}/{s['total']} · near-misses {s['near']}"
    )
    parts = " · ".join(
        f"{d} {s['by_diff'][d]['passed']}/{s['by_diff'][d]['run']}" for d in _DIFFS
    )
    lines.append(f"By difficulty: {parts}")
    if s["run"]:
        lines.append(
            f"Cost: {s['tokens']:,} tokens · {s['recovered']} self-corrected · "
            f"avg {s['avg_elapsed']:.1f}s/question"
        )
    return "\n".join(lines)


def render_markdown(rows: list[dict], meta: dict) -> str:
    s = summarize(rows)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = []
    out.append("# Eval scorecard — Data-Analysis Agent\n")
    out.append(
        f"**Model:** `{meta.get('model')}` · **tag:** `{meta.get('tag')}` · "
        f"**tolerance:** ±{meta.get('tol_pct')}% · **generated:** {ts}\n"
    )
    out.append(
        f"## {s['passed']}/{s['run']} correct ({s['pct']:.0f}%)  ·  "
        f"coverage {s['run']}/{s['total']}\n"
    )
    if meta.get("stopped") == "rate_limit":
        out.append(
            "> ⏳ A run stopped on the free-tier daily limit. Cached results are "
            "kept — re-run the same command later to fill in the rest.\n"
        )

    out.append("### By difficulty\n")
    out.append("| Difficulty | Passed / Run | Total |")
    out.append("|---|---|---|")
    for d in _DIFFS:
        b = s["by_diff"][d]
        out.append(f"| {d} | {b['passed']} / {b['run']} | {b['total']} |")
    out.append("")

    if s["run"]:
        out.append(
            f"**Behavior:** {s['tokens']:,} total tokens · {s['recovered']} answers "
            f"needed self-correction · avg {s['avg_elapsed']:.1f}s/question.\n"
        )

    out.append("### Per-question\n")
    out.append("| | id | diff | expected | agent answer | note |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        sym = _SYMBOL[r["verdict"]]
        exp = _short(str(r["expected"]), 40)
        got = _short(r.get("answer") or "—", 60)
        note = _short(r["detail"], 50)
        out.append(f"| {sym} | {r['id']} | {r['difficulty']} | {exp} | {got} | {note} |")
    out.append("")

    pending = [r["id"] for r in rows if r["verdict"] == "pending"]
    if pending:
        out.append(f"**Not yet run ({len(pending)}):** {', '.join(pending)}\n")
    out.append(
        "_Numeric answers pass within the tolerance above; a 🟡 near-miss is often "
        "the gross-vs-net revenue definition, not a reasoning error. Categorical "
        "answers need an exact word match; dict answers need every value present._\n"
    )
    return "\n".join(out)


def _short(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"
