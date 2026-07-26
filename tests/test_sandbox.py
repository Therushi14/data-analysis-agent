"""Integration tests for the sandbox (real subprocess, no network / no API key).

Covers the plan.md §7 acceptance criteria: success, dataframe result, error
traceback, disallowed import, timeout, and figure capture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent.tools.sandbox import Sandbox


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3, 4], "g": ["a", "a", "b", "b"]})


def _sandbox(tmp_path, **kw) -> Sandbox:
    sb = Sandbox(work_dir=tmp_path, **kw)
    return sb


def test_scalar_success(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("result = int(df['x'].sum())")
    assert r.ok
    assert r.result_kind == "scalar"
    assert "10" in (r.result_repr or "")


def test_dataframe_result(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("result = df.groupby('g')['x'].sum().reset_index()")
    assert r.ok
    assert r.result_kind == "dataframe"
    assert r.dataframe_preview is not None


def test_stdout_captured(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("print('hello from sandbox')")
    assert r.ok
    assert "hello from sandbox" in r.stdout


def test_error_traceback(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("result = df['nonexistent_col']")
    assert not r.ok
    assert r.error_traceback is not None
    assert "KeyError" in r.error_traceback


def test_disallowed_import(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("import os\nresult = os.getcwd()")
    assert not r.ok
    assert "blocked by the sandbox policy" in (r.error_traceback or "")


def test_timeout(tmp_path, df):
    sb = _sandbox(tmp_path, timeout_s=3)
    sb.prepare_data(df)
    r = sb.run("while True:\n    pass")
    assert not r.ok
    assert r.timed_out


def test_figure_capture(tmp_path, df):
    sb = _sandbox(tmp_path)
    sb.prepare_data(df)
    r = sb.run("plt.plot(df['x'])\nresult = 'plotted'")
    assert r.ok
    assert r.figure_path is not None
    assert Path(r.figure_path).exists()


def test_relative_work_dir_is_resolved(tmp_path, df, monkeypatch):
    # A relative work_dir must still work: the worker's cwd is the work_dir, so
    # unresolved relative --data/--code/--out paths would double-nest and 404.
    monkeypatch.chdir(tmp_path)
    sb = Sandbox(work_dir="relwork")
    assert sb.work_dir.is_absolute()
    sb.prepare_data(df)
    r = sb.run("result = int(df['x'].sum())")
    assert r.ok
    assert "10" in (r.result_repr or "")
