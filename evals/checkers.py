"""Compare a natural-language agent answer against verified ground truth.

The agent replies in prose ("total net revenue is $28,448,095.67"), so we can't
just `float(answer)`. Instead we extract candidate numbers (handling `$`, commas,
`%`, and month names) and match within a tolerance; categorical answers match by
word; dict answers require every expected value to appear. All deterministic and
offline — no extra LLM call, so scoring spends no quota.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# A number, optionally $-prefixed, with thousands separators and/or decimals.
_NUM_RE = re.compile(r"[-+]?\$?\s?(\d[\d,]*(?:\.\d+)?)")


@dataclass
class CheckResult:
    ok: bool
    verdict: str      # "pass" | "near" | "fail"
    detail: str
    matched: int = 0  # for dict: how many expected values were found
    total: int = 0    # for dict: how many were expected


def extract_numbers(text: str) -> list[float]:
    """Every numeric value mentioned in the text, plus any month names as 1–12."""
    nums: list[float] = []
    for raw in _NUM_RE.findall(text or ""):
        try:
            nums.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    low = (text or "").lower()
    for name, n in _MONTHS.items():
        if name in low:
            nums.append(float(n))
    return nums


def _rel_err_pct(got: float, exp: float) -> float:
    if exp == 0:
        return 0.0 if abs(got) < 1e-6 else math.inf
    return abs(got - exp) / abs(exp) * 100.0


def check_numeric(text: str, expected: float, tol_pct: float, near_pct: float = 5.0) -> CheckResult:
    nums = extract_numbers(text)
    if not nums:
        return CheckResult(False, "fail", "no number found in answer")
    best = min(nums, key=lambda n: _rel_err_pct(n, expected))
    err = _rel_err_pct(best, expected)
    if err <= tol_pct:
        return CheckResult(True, "pass", f"matched {best:g} (±{err:.2f}%)")
    if err <= near_pct:
        return CheckResult(False, "near", f"closest {best:g} off by {err:.2f}% (>{tol_pct}%)")
    return CheckResult(False, "fail", f"closest {best:g} off by {err:.2f}%")


def check_categorical(text: str, expected: str) -> CheckResult:
    exp = str(expected).strip()
    # whole-token match so "East" doesn't match inside "Eastern", but multi-word
    # names ("David Okafor", 'Monitor 27"') still match as a phrase.
    pattern = r"(?<!\w)" + re.escape(exp) + r"(?!\w)"
    if re.search(pattern, text or "", re.IGNORECASE):
        return CheckResult(True, "pass", f"found '{exp}'")
    return CheckResult(False, "fail", f"'{exp}' not present in answer")


def check_dict(text: str, expected: dict, tol_pct: float) -> CheckResult:
    nums = extract_numbers(text)
    total = len(expected)
    matched = 0
    missing = []
    for key, val in expected.items():
        if any(_rel_err_pct(n, float(val)) <= tol_pct for n in nums):
            matched += 1
        else:
            missing.append(f"{key}={val}")
    ok = matched == total
    verdict = "pass" if ok else ("near" if matched >= math.ceil(total / 2) else "fail")
    detail = f"matched {matched}/{total} values"
    if missing:
        detail += " · missing: " + ", ".join(missing[:4])
    return CheckResult(ok, verdict, detail, matched=matched, total=total)


def check_item(item: dict, answer_text: str, tol_pct: float = 0.5) -> CheckResult:
    """Route one eval item to the checker for its `kind`."""
    kind = item["kind"]
    expected = item["expected"]
    if kind == "numeric":
        return check_numeric(answer_text, float(expected), tol_pct)
    if kind == "categorical":
        return check_categorical(answer_text, expected)
    if kind == "dict":
        return check_dict(answer_text, expected, tol_pct)
    return CheckResult(False, "fail", f"unknown kind '{kind}'")
