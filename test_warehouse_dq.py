
import os
from pathlib import Path
import pandas as pd

# Allow overriding warehouse dir via env var; default to ./warehouse
WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", Path.cwd() / "warehouse")) / "csv"

def _load(name: str) -> pd.DataFrame:
    path = WAREHOUSE_DIR / f"{name}.csv"
    assert path.exists(), f"Missing expected output: {path}"
    return pd.read_csv(path)

def test_dq_all_pass():
    dq = _load("dq_results")
    assert "status" in dq.columns, "dq_results must have a status column"
    failing = dq[dq["status"] != "PASS"]
    assert failing.empty, f"Data Quality failures:\n{failing.to_string(index=False)}"

def test_fk_fact_play_session_user():
    fps = _load("fact_play_session")
    du = _load("dim_user")[["user_id"]].drop_duplicates()
    merged = fps.merge(du, on="user_id", how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    assert missing.empty, f"Missing user_id(s) in dim_user:\n{missing['user_id'].dropna().unique()}"

def test_fk_fact_user_plan_plan():
    fup = _load("fact_user_plan")
    dp = _load("dim_plan")[["plan_id"]].drop_duplicates()
    merged = fup.merge(dp, on="plan_id", how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    assert missing.empty, f"Missing plan_id(s) in dim_plan:\n{missing['plan_id'].dropna().unique()}"

def test_sessions_by_channel_exists():
    df = _load("insight_sessions_by_channel")
    assert {"channel_group", "session_count"}.issubset(df.columns)
    assert len(df) >= 1
