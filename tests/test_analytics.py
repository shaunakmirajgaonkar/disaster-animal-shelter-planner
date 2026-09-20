import pandas as pd
from analytics import classify, enrich, markdown_table, scenario_adjust


def sample_df():
    return pd.DataFrame([
        ["1","North","Shelter A","Dogs",30,"High",20,20,50,35,400,60,12,8,40,2,0,30],
        ["2","South","Shelter B","Cattle",40,"Critical",18,40,45,42,500,120,6,5,25,1,2,120],
    ], columns=[
        "record_id","zone","site_name","animal_type","animal_count","evacuation_priority","transport_capacity","travel_minutes","facility_capacity","current_booked","water_capacity_l_day","feed_capacity_kg_day","backup_power_hours","handling_staff","crate_pen_availability","veterinary_support","livestock_support","last_inspection_days"
    ])


def test_enrich_outputs_expected_fields():
    out = enrich(sample_df())
    for col in ["shelter_pressure_score","review_priority","dominant_driver","capacity_gap","transport_gap","supply_days"]:
        assert col in out.columns
    assert out["shelter_pressure_score"].between(0,100).all()


def test_classification_thresholds():
    assert classify(20) == "Low"
    assert classify(40) == "Moderate"
    assert classify(60) == "High"
    assert classify(80) == "Critical"


def test_scenario_reduces_pressure_with_added_capacity_and_transport():
    base = enrich(sample_df())
    scen = scenario_adjust(sample_df(), capacity_change=30, transport_change=25, staff_change=3, travel_change_pct=-20, supply_days_change=2)
    assert scen["shelter_pressure_score"].mean() <= base["shelter_pressure_score"].mean()


def test_markdown_table_does_not_need_tabulate():
    text = markdown_table(enrich(sample_df())[["site_name","shelter_pressure_score","review_priority"]])
    assert "| site_name |" in text
    assert "Critical" in text or "High" in text
