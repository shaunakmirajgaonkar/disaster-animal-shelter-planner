from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "record_id",
    "zone",
    "site_name",
    "animal_type",
    "animal_count",
    "evacuation_priority",
    "transport_capacity",
    "travel_minutes",
    "facility_capacity",
    "current_booked",
    "water_capacity_l_day",
    "feed_capacity_kg_day",
    "backup_power_hours",
    "handling_staff",
    "crate_pen_availability",
    "veterinary_support",
    "livestock_support",
    "last_inspection_days",
]

NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in {"record_id", "zone", "site_name", "animal_type", "evacuation_priority"}]


def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["record_id", "zone", "site_name", "animal_type", "evacuation_priority"]:
        out[col] = out[col].astype(str).str.strip()
    out["animal_count"] = out["animal_count"].clip(lower=0)
    out["transport_capacity"] = out["transport_capacity"].clip(lower=0)
    out["travel_minutes"] = out["travel_minutes"].clip(lower=0)
    out["facility_capacity"] = out["facility_capacity"].clip(lower=0)
    out["current_booked"] = out["current_booked"].clip(lower=0)
    out["last_inspection_days"] = out["last_inspection_days"].clip(lower=0)
    return out


def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)


def classify(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_input(df)
    if out.empty:
        return out.assign(
            occupancy_pct=pd.Series(dtype=float),
            capacity_gap=pd.Series(dtype=float),
            transport_gap=pd.Series(dtype=float),
            supply_days=pd.Series(dtype=float),
            shelter_pressure_score=pd.Series(dtype=float),
            review_priority=pd.Series(dtype=str),
            dominant_driver=pd.Series(dtype=str),
            inspection_pressure=pd.Series(dtype=float),
        )

    out["occupancy_pct"] = np.where(
        out["facility_capacity"] > 0,
        100 * out["current_booked"] / out["facility_capacity"],
        100,
    )
    out["capacity_gap"] = (out["animal_count"] - (out["facility_capacity"] - out["current_booked"])).clip(lower=0)
    out["transport_gap"] = (out["animal_count"] - out["transport_capacity"]).clip(lower=0)
    out["supply_days"] = np.minimum(
        np.where(out["animal_count"] > 0, out["water_capacity_l_day"] / np.maximum(out["animal_count"], 1), 99),
        np.where(out["animal_count"] > 0, out["feed_capacity_kg_day"] / np.maximum(out["animal_count"], 1), 99),
    )
    out["supply_days"] = out["supply_days"].clip(0, 99)

    # Higher values indicate greater operational planning pressure.
    priority_map = {"Routine": 0.15, "Elevated": 0.45, "High": 0.75, "Critical": 1.0}
    priority_pressure = out["evacuation_priority"].map(priority_map).fillna(0.45)
    occupancy_pressure = (out["occupancy_pct"] / 100).clip(0, 1)
    capacity_pressure = _norm(out["capacity_gap"])
    transport_pressure = _norm(out["transport_gap"])
    travel_pressure = _norm(out["travel_minutes"])
    supply_pressure = (1 - (out["supply_days"] / 7).clip(0, 1))
    maintenance_pressure = (out["last_inspection_days"] / 180).clip(0, 1)
    support_pressure = 1 - (
        0.35 * (out["veterinary_support"] / 2).clip(0, 1)
        + 0.35 * (out["livestock_support"] / 2).clip(0, 1)
        + 0.30 * (out["handling_staff"] / 12).clip(0, 1)
    )

    score = 100 * (
        0.22 * occupancy_pressure
        + 0.20 * capacity_pressure
        + 0.17 * transport_pressure
        + 0.12 * travel_pressure
        + 0.11 * supply_pressure
        + 0.07 * maintenance_pressure
        + 0.06 * support_pressure
        + 0.05 * priority_pressure
    )
    out["shelter_pressure_score"] = score.clip(0, 100).round(1)
    out["review_priority"] = out["shelter_pressure_score"].map(classify)

    driver_components = pd.DataFrame(
        {
            "Capacity": capacity_pressure,
            "Transport": transport_pressure,
            "Occupancy": occupancy_pressure,
            "Travel": travel_pressure,
            "Supplies": supply_pressure,
            "Maintenance": maintenance_pressure,
            "Support": support_pressure,
        },
        index=out.index,
    )
    out["dominant_driver"] = driver_components.idxmax(axis=1)
    out["inspection_pressure"] = maintenance_pressure.round(3)
    return out


def summarize(df: pd.DataFrame) -> dict:
    x = enrich(df)
    return {
        "sites": int(x["site_name"].nunique()) if not x.empty else 0,
        "zones": int(x["zone"].nunique()) if not x.empty else 0,
        "animals": int(x["animal_count"].sum()) if not x.empty else 0,
        "available_capacity": int((x["facility_capacity"] - x["current_booked"]).clip(lower=0).sum()) if not x.empty else 0,
        "high_priority": int(x["review_priority"].isin(["High", "Critical"]).sum()) if not x.empty else 0,
        "avg_score": float(x["shelter_pressure_score"].mean()) if not x.empty else 0.0,
    }


def scenario_adjust(df: pd.DataFrame, capacity_change: float = 0, transport_change: float = 0,
                     staff_change: float = 0, travel_change_pct: float = 0,
                     supply_days_change: float = 0, inspection_days_change: float = 0) -> pd.DataFrame:
    x = clean_input(df)
    x = x.copy()
    x["facility_capacity"] = (x["facility_capacity"] + capacity_change).clip(lower=0)
    x["transport_capacity"] = (x["transport_capacity"] + transport_change).clip(lower=0)
    x["handling_staff"] = (x["handling_staff"] + staff_change).clip(lower=0)
    x["travel_minutes"] = (x["travel_minutes"] * (1 + travel_change_pct / 100)).clip(lower=0)
    x["water_capacity_l_day"] = (x["water_capacity_l_day"] + supply_days_change * x["animal_count"]).clip(lower=0)
    x["feed_capacity_kg_day"] = (x["feed_capacity_kg_day"] + supply_days_change * x["animal_count"]).clip(lower=0)
    x["last_inspection_days"] = (x["last_inspection_days"] + inspection_days_change).clip(lower=0)
    return enrich(x)


def markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    view = df.head(max_rows).copy()
    if view.empty:
        return "_No records available._"
    cols = list(view.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for val in row.tolist():
            s = "" if pd.isna(val) else str(val)
            s = s.replace("|", "\\|").replace("\n", " ")
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
