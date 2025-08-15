from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.resolve()
RAW_DIR = Path(os.getenv("DATA_DIR", BASE_DIR))               
WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", RAW_DIR / "warehouse"))
(WAREHOUSE_DIR / "csv").mkdir(parents=True, exist_ok=True)
(WAREHOUSE_DIR / "parquet").mkdir(parents=True, exist_ok=True)

# Helpers

KNOWN_DT_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%d-%b-%Y",
]

def parse_dt(series: pd.Series) -> pd.Series:

    s = series.astype(str)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    parsed_any = False
    for fmt in KNOWN_DT_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            mask = parsed.notna() & out.isna()
            if mask.any():
                out.loc[mask] = parsed.loc[mask]
                parsed_any = True
        except Exception:
            pass
    if not parsed_any:
        out = pd.to_datetime(s, format="mixed", errors="coerce")
    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out

def snake(s: str) -> str:
    s = re.sub(r"[/\s\-]+", "_", s.strip())
    s = re.sub(r"__+", "_", s)
    return s.lower()

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [snake(c) for c in df.columns]
    for col in df.columns:
        if any(k in col for k in ["date", "datetime", "time", "expiry"]):
            df[col] = parse_dt(df[col])
    return df

def to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def to_float_currency(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                    .str.replace(r"[^0-9\.\-]", "", regex=True)
                    .replace({"": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def write_table(name: str, df: pd.DataFrame):
    csv_path = WAREHOUSE_DIR / "csv" / f"{name}.csv"
    pq_path = WAREHOUSE_DIR / "parquet" / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass
    return csv_path, pq_path

def select_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# Load raw

files = {
    "user": RAW_DIR / "user.csv",
    "user_registration": RAW_DIR / "user_registration.csv",
    "user_plan": RAW_DIR / "user_plan.csv",
    "user_payment_detail": RAW_DIR / "user_payment_detail.csv",
    "user_play_session": RAW_DIR / "user_play_session.csv",
    "status_code": RAW_DIR / "status_code.csv",
    "plan_payment_frequency": RAW_DIR / "plan_payment_frequency.csv",
    "plan": RAW_DIR / "plan.csv",
    "channel_code": RAW_DIR / "channel_code.csv",
}
raw = {k: load_csv(p) for k, p in files.items()}

# Normalize types

user = to_int(raw["user"].copy(), ["user_id"])
ureg = to_int(raw["user_registration"].copy(), ["user_registration_id", "user_id"])

plan = to_int(raw["plan"].copy(), ["plan_id"])
plan = to_float_currency(plan, ["cost_amount"])
if "payment_frequency_code" not in plan.columns:
    for alt in ["payment_frequency", "payment_freq_code", "frequency_code"]:
        if alt in plan.columns:
            plan = plan.rename(columns={alt: "payment_frequency_code"})
            break

pf = raw["plan_payment_frequency"].copy()
if "payment_frequency_code" not in pf.columns:
    for alt in ["frequency_code", "payment_frequency"]:
        if alt in pf.columns:
            pf = pf.rename(columns={alt: "payment_frequency_code"})
            break

uplan = raw["user_plan"].copy()
if "plan_id" in uplan.columns:
    uplan["plan_id"] = pd.to_numeric(uplan["plan_id"], errors="coerce").astype("Int64")
uplan = to_int(uplan, ["user_registration_id", "payment_detail_id"])

upay = to_int(raw["user_payment_detail"].copy(), ["payment_detail_id"])

ps = to_int(raw["user_play_session"].copy(), ["play_session_id", "user_id", "total_score"])

status = raw["status_code"].copy()
if "status_code" not in status.columns:
    for alt in ["play_session_status_code", "code"]:
        if alt in status.columns:
            status = status.rename(columns={alt: "status_code"})
            break

channel = raw["channel_code"].copy()
if "channel_code" not in channel.columns:
    for alt in ["play_session_channel_code", "code"]:
        if alt in channel.columns:
            channel = channel.rename(columns={alt: "channel_code"})
            break

# Dimensions

uids = pd.Series(dtype="Int64")
if "user_id" in user.columns: uids = pd.concat([uids, user["user_id"]])
if "user_id" in ureg.columns: uids = pd.concat([uids, ureg["user_id"]])
if "user_id" in ps.columns:   uids = pd.concat([uids, ps["user_id"]])

all_user_ids = (
    pd.DataFrame({"user_id": pd.to_numeric(uids, errors="coerce").astype("Int64")})
    .dropna(subset=["user_id"]).drop_duplicates()
)

user_attr_cols = ["user_id", "ip_address", "social_media_handle", "email"]
dim_user = all_user_ids.merge(
    user[user_attr_cols] if set(user_attr_cols).issubset(user.columns) else user[["user_id"]],
    on="user_id", how="left"
).sort_values("user_id").reset_index(drop=True)

for c in user_attr_cols:
    if c not in dim_user.columns:
        dim_user[c] = pd.NA

sess_uids = set(pd.to_numeric(ps["user_id"], errors="coerce").dropna().astype("Int64").unique()) if "user_id" in ps.columns else set()
dim_uids  = set(dim_user["user_id"].dropna().unique()) if "user_id" in dim_user.columns else set()
missing_uids = sorted(list(sess_uids - dim_uids))
if missing_uids:
    stub = pd.DataFrame({
        "user_id": pd.Series(missing_uids, dtype="Int64"),
        "ip_address": pd.NA,
        "social_media_handle": pd.NA,
        "email": pd.NA
    })
    dim_user = (
        pd.concat([dim_user, stub], ignore_index=True)
          .drop_duplicates(subset=["user_id"])
          .sort_values("user_id")
          .reset_index(drop=True)
    )

dim_user_registration = (
    ureg[select_existing(ureg, ["user_registration_id", "user_id", "username", "email", "first_name", "last_name"])]
    .drop_duplicates()
    .reset_index(drop=True)
)

dim_payment_frequency = (
    pf[select_existing(pf, ["payment_frequency_code", "english_description", "french_description"])]
    .drop_duplicates()
    .reset_index(drop=True)
)

dim_plan = plan.merge(
    dim_payment_frequency[select_existing(dim_payment_frequency, ["payment_frequency_code", "english_description", "french_description"])],
    on="payment_frequency_code", how="left", suffixes=("", "_freq")
)
dim_plan = dim_plan[select_existing(dim_plan, ["plan_id", "payment_frequency_code", "cost_amount", "english_description", "french_description"])]

dim_channel = (
    channel[select_existing(channel, ["channel_code", "english_description", "french_description"])]
    .drop_duplicates()
    .reset_index(drop=True)
)

dim_status = (
    status[select_existing(status, ["status_code", "english_description", "french_description"])]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Facts

fact_play_session = ps[select_existing(ps, [
    "play_session_id", "user_id", "start_datetime", "end_datetime",
    "channel_code", "status_code", "total_score"
])].copy()

if "start_datetime" in fact_play_session.columns and "end_datetime" in fact_play_session.columns:
    fact_play_session["duration_seconds"] = (
        (fact_play_session["end_datetime"] - fact_play_session["start_datetime"]).dt.total_seconds()
    )

fact_user_plan = uplan[select_existing(uplan, ["user_registration_id", "payment_detail_id", "plan_id", "start_date", "end_date"])].copy()
fact_payment_detail = upay[select_existing(upay, ["payment_detail_id", "payment_method_code", "payment_method_value", "payment_method_expiry"])].copy()

# Data Quality

dq_results = []

def dq_check_no_nulls(df, cols, name):
    for c in cols:
        if c in df.columns:
            n = int(df[c].isna().sum())
            dq_results.append({"check": f"{name}: {c} not null", "status": "PASS" if n == 0 else "FAIL", "detail": f"null_count={n}"})

def dq_check_fk(parent_df, parent_col, child_df, child_col, name):
    if parent_col in parent_df.columns and child_col in child_df.columns:
        missing = set(child_df[child_col].dropna().unique()) - set(parent_df[parent_col].dropna().unique())
        dq_results.append({"check": f"FK {name}: {child_col} in child exists in parent {parent_col}", "status": "PASS" if len(missing) == 0 else "FAIL", "detail": f"missing_keys_count={len(missing)}"})

def dq_check_date_order(df, start_col, end_col, name):
    if start_col in df.columns and end_col in df.columns:
        bad = df[(df[start_col].notna()) & (df[end_col].notna()) & (df[start_col] > df[end_col])]
        dq_results.append({"check": f"{name}: {start_col} <= {end_col}", "status": "PASS" if len(bad) == 0 else "FAIL", "detail": f"violations={len(bad)}"})

dq_check_no_nulls(dim_user, ["user_id"], "dim_user")
dq_check_no_nulls(dim_user_registration, ["user_registration_id", "user_id"], "dim_user_registration")
dq_check_no_nulls(dim_plan, ["plan_id", "payment_frequency_code"], "dim_plan")
dq_check_no_nulls(dim_payment_frequency, ["payment_frequency_code"], "dim_payment_frequency")
dq_check_no_nulls(dim_channel, ["channel_code"], "dim_channel")
dq_check_no_nulls(dim_status, ["status_code"], "dim_status")
dq_check_no_nulls(fact_play_session, ["play_session_id", "user_id", "channel_code", "status_code"], "fact_play_session")
dq_check_no_nulls(fact_user_plan, ["user_registration_id", "plan_id"], "fact_user_plan")
dq_check_no_nulls(fact_payment_detail, ["payment_detail_id"], "fact_payment_detail")

dq_check_fk(dim_user, "user_id", fact_play_session, "user_id", "fact_play_session -> dim_user")
dq_check_fk(dim_channel, "channel_code", fact_play_session, "channel_code", "fact_play_session -> dim_channel")
dq_check_fk(dim_status, "status_code", fact_play_session, "status_code", "fact_play_session -> dim_status")
dq_check_fk(dim_plan, "plan_id", fact_user_plan, "plan_id", "fact_user_plan -> dim_plan")
dq_check_fk(dim_user_registration, "user_registration_id", fact_user_plan, "user_registration_id", "fact_user_plan -> dim_user_registration")
dq_check_fk(fact_payment_detail, "payment_detail_id", fact_user_plan, "payment_detail_id", "fact_user_plan -> fact_payment_detail")

dq_check_date_order(fact_user_plan, "start_date", "end_date", "fact_user_plan")
dq_check_date_order(fact_play_session, "start_datetime", "end_datetime", "fact_play_session")

dq_df = pd.DataFrame(dq_results)

# Insights

def classify_channel(desc: str | float) -> str:
    if pd.isna(desc):
        return "Unknown"
    s = str(desc).lower()
    if any(k in s for k in ["mobile", "app", "ios", "android"]):
        return "Mobile App"
    if any(k in s for k in ["web", "online", "browser"]):
        return "Online"
    return "Other"

desc_col_chan = next((c for c in ["english_description", "french_description"] if c in dim_channel.columns), None)
fps = fact_play_session.copy()
if desc_col_chan and "channel_code" in fps.columns:
    chan_map = dim_channel.set_index("channel_code")[desc_col_chan]
    fps["channel_desc"] = fps["channel_code"].map(chan_map)
    fps["channel_group"] = fps["channel_desc"].apply(classify_channel)
else:
    fps["channel_group"] = fps.get("channel_code", pd.Series(dtype="object"))

sessions_by_channel = fps.groupby("channel_group", dropna=False)["play_session_id"].count().reset_index(name="session_count")

pf_desc_col = next((c for c in ["english_description", "french_description"] if c in dim_payment_frequency.columns), None)
plan_with_freq = dim_plan.merge(dim_payment_frequency, on="payment_frequency_code", suffixes=("_plan", "_freq"), how="left")

desc_col_merged = None
if pf_desc_col:
    for cand in [f"{pf_desc_col}_freq", f"{pf_desc_col}_plan", pf_desc_col]:
        if cand in plan_with_freq.columns:
            desc_col_merged = cand
            break

def classify_frequency(desc: str | float) -> str:
    if pd.isna(desc):
        return "Unknown"
    s = str(desc).lower()
    if re.search(r"(?:once|one[-\s]?time|onetime|single)", s):
        return "One-time"
    if re.search(r"(?:month|monthly|year|annual|annually|week|weekly|day|daily)", s):
        return "Subscription"
    return "Subscription"

if desc_col_merged:
    plan_with_freq["billing_type"] = plan_with_freq[desc_col_merged].apply(classify_frequency)
else:
    plan_with_freq["billing_type"] = "Unknown"

uplan_with_type = fact_user_plan.merge(plan_with_freq[["plan_id", "billing_type"]], on="plan_id", how="left")
adoption = uplan_with_type.groupby("billing_type")["user_registration_id"].nunique().reset_index(name="unique_users")

# Revenue 2024
START_2024 = pd.Timestamp("2024-01-01")
END_2024 = pd.Timestamp("2024-12-31 23:59:59")
days_in_2024 = (END_2024 - START_2024).days + 1

def period_multiplier(desc: str | float) -> float:
    if pd.isna(desc): return np.nan
    s = str(desc).lower()
    if "month" in s: return 12.0
    if "year" in s or "annual" in s: return 1.0
    if "week" in s: return 52.0
    if "day" in s: return 365.0
    if re.search(r"(?:once|one[-\s]?time|onetime|single)", s): return 0.0
    return 12.0

pf_mult = dim_payment_frequency.copy()
if pf_desc_col:
    pf_mult["periods_per_year"] = pf_mult[pf_desc_col].apply(period_multiplier)
else:
    pf_mult["periods_per_year"] = 12.0

plan_rev = dim_plan.merge(pf_mult[["payment_frequency_code", "periods_per_year"]], on="payment_frequency_code", how="left")
upl = fact_user_plan.merge(plan_rev[["plan_id", "cost_amount", "payment_frequency_code", "periods_per_year"]], on="plan_id", how="left")

for c in ["start_date", "end_date"]:
    if c in upl.columns:
        upl[c] = parse_dt(upl[c])
upl["start_date"] = upl["start_date"].fillna(pd.Timestamp.min)
upl["end_date"] = upl["end_date"].fillna(pd.Timestamp.max)

upl["overlap_start"] = upl["start_date"].clip(lower=START_2024)
upl["overlap_end"] = upl["end_date"].clip(upper=END_2024)
upl["overlap_days"] = (upl["overlap_end"] - upl["overlap_start"]).dt.days.clip(lower=0)

fr = dim_payment_frequency.copy()
if pf_desc_col:
    fr["is_onetime_desc"] = fr[pf_desc_col].str.contains(r"(?:once|one[-\s]?time|onetime|single)", case=False, regex=True, na=False)
else:
    fr["is_onetime_desc"] = False

upl = upl.merge(fr[["payment_frequency_code", "is_onetime_desc"]], on="payment_frequency_code", how="left")
upl["is_onetime"] = upl["is_onetime_desc"].fillna(False)

upl["recurring_rev_2024"] = (
    upl.loc[~upl["is_onetime"], "cost_amount"] *
    (upl.loc[~upl["is_onetime"], "overlap_days"] / days_in_2024) *
    upl.loc[~upl["is_onetime"], "periods_per_year"].fillna(12.0)
)
upl["onetime_rev_2024"] = np.where(
    upl["is_onetime"] & upl["start_date"].between(START_2024, END_2024, inclusive="both"),
    upl["cost_amount"],
    0.0
)
upl["gross_revenue_2024"] = upl["recurring_rev_2024"].fillna(0.0) + upl["onetime_rev_2024"].fillna(0.0)
gross_revenue_2024 = float(upl["gross_revenue_2024"].sum()) if len(upl) else 0.0

# Write outputs

outputs = {}
for name, df in [
    ("dim_user", dim_user),
    ("dim_user_registration", dim_user_registration),
    ("dim_payment_frequency", dim_payment_frequency),
    ("dim_plan", dim_plan),
    ("dim_channel", dim_channel),
    ("dim_status", dim_status),
    ("fact_play_session", fact_play_session),
    ("fact_user_plan", fact_user_plan),
    ("fact_payment_detail", fact_payment_detail),
    ("dq_results", dq_df),
]:
    outputs[name] = write_table(name, df)[0]

sessions_by_channel = sessions_by_channel.sort_values("session_count", ascending=False)
adoption = adoption.sort_values("unique_users", ascending=False)

sessions_by_channel.to_csv(WAREHOUSE_DIR / "csv" / "insight_sessions_by_channel.csv", index=False)
adoption.to_csv(WAREHOUSE_DIR / "csv" / "insight_plan_adoption.csv", index=False)

print("Warehouse CSV outputs:")
for k, v in outputs.items():
    print(f"- {k}: {v}")
print("\nInsights CSV:")
print(f"- sessions_by_channel: {WAREHOUSE_DIR / 'csv' / 'insight_sessions_by_channel.csv'}")
print(f"- plan_adoption: {WAREHOUSE_DIR / 'csv' / 'insight_plan_adoption.csv'}")
print(f"\nGross Revenue for 2024 (approx.): {gross_revenue_2024:,.2f}")
