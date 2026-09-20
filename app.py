from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import REQUIRED_COLUMNS, enrich, markdown_table, scenario_adjust, summarize, validate_columns

st.set_page_config(
    page_title="Disaster Animal Shelter Planner",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --ink:#17233b; --muted:#667085; --line:#e7ecf3; --panel:#ffffff; --bg:#f5f8fc; }
.stApp { background: linear-gradient(145deg,#f8fbff 0%,#eef8f4 48%,#fdf7ee 100%); color:var(--ink); }
.block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; }
.hero { background: linear-gradient(120deg,#ffffff 0%,#edf7ff 52%,#f5fffa 100%); border:1px solid var(--line); border-radius:26px; padding:24px 28px; box-shadow:0 14px 35px rgba(33,52,79,.08); margin-bottom:18px; }
.hero h1 { margin:0; color:#17324d; font-size:2.25rem; letter-spacing:-.02em; }
.hero p { margin:.45rem 0 0; color:#536174; font-size:1rem; }
.badge { display:inline-block; padding:7px 12px; border-radius:999px; font-weight:700; font-size:.78rem; margin-bottom:9px; }
.badge-local { background:#e7f7ef; color:#136f48; }
.metric { background:#fff; border:1px solid var(--line); border-radius:19px; padding:16px 18px; box-shadow:0 8px 24px rgba(33,52,79,.06); min-height:118px; }
.metric .label { color:#667085; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
.metric .value { color:#16263e; font-size:1.72rem; font-weight:800; margin-top:5px; }
.metric .sub { color:#7d8798; font-size:.8rem; margin-top:5px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:20px; padding:18px 20px; box-shadow:0 8px 24px rgba(33,52,79,.055); }
.small-note { color:#667085; font-size:.84rem; }
.priority-pill { padding:5px 9px; border-radius:999px; font-weight:700; font-size:.75rem; }
.stButton>button { border-radius:12px; font-weight:700; border:1px solid #d9e2ec; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e5eaf0; }
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def priority_badge(level: str) -> str:
    mapping = {"Critical": ("#fde8e8", "#b42318"), "High": ("#fff0e1", "#b54708"), "Moderate": ("#fff8d6", "#946200"), "Low": ("#e9f8ee", "#137333")}
    bg, fg = mapping.get(level, ("#eef2f6", "#475467"))
    return f'<span class="priority-pill" style="background:{bg};color:{fg}">{level}</span>'


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "sample_disaster_animal_shelter.csv"
    return pd.read_csv(path)


with st.sidebar:
    st.markdown("### 🐾 Shelter Control Center")
    st.caption("100% local • CSV-driven • planning support")
    uploaded = st.file_uploader("Upload shelter planning CSV", type=["csv"], help="CSV must contain all required shelter-planning fields.")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Shelter Capacity",
            "Evacuation & Transport",
            "Animal Mix",
            "Facilities & Support",
            "Priority Queue",
            "Scenario Lab",
            "Reports & Export",
            "Data Explorer",
        ],
        index=0,
    )
    st.divider()
    st.caption("Required columns")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    raw = load_sample()

missing = validate_columns(raw)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.info("Upload the provided template or the bundled sample CSV.")
    st.stop()

df = enrich(raw)
summary = summarize(raw)

st.markdown(
    '<div class="hero"><div class="badge badge-local">LOCAL-FIRST • DISASTER ANIMAL SHELTER PLANNING</div>'
    '<h1>🐾 Disaster Animal Shelter Planner</h1>'
    '<p>Plan temporary shelter capacity for pets and livestock during disasters using evacuation exposure, animal counts, transport, facility capacity, supplies, staffing, and support readiness.</p></div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    c = st.columns(5)
    with c[0]: metric_card("Sites", f"{summary['sites']}", "temporary shelter locations")
    with c[1]: metric_card("Animals", f"{summary['animals']:,}", "animals in planning records")
    with c[2]: metric_card("Open Capacity", f"{summary['available_capacity']:,}", "unbooked facility spaces")
    with c[3]: metric_card("Review Queue", f"{summary['high_priority']}", "High/Critical sites")
    with c[4]: metric_card("Avg Pressure", f"{summary['avg_score']:.1f}/100", "screening signal")
    st.write("")
    left, right = st.columns([1.15, .85])
    with left:
        st.markdown('<div class="panel"><h3>Capacity pressure by site</h3>', unsafe_allow_html=True)
        site = df.groupby(["site_name", "zone"], as_index=False).agg(
            animals=("animal_count", "sum"),
            available=("capacity_gap", lambda s: 0),
            score=("shelter_pressure_score", "mean"),
        )
        site["available"] = (
            df.assign(available_slots=(df["facility_capacity"] - df["current_booked"]).clip(lower=0))
            .groupby("site_name")["available_slots"].sum().reindex(site["site_name"]).to_numpy()
        )
        fig = px.bar(site.sort_values("score", ascending=False), x="site_name", y="score", color="score", color_continuous_scale="Tealgrn", labels={"score":"Pressure score","site_name":"Shelter site"})
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=15,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>Priority distribution</h3>', unsafe_allow_html=True)
        counts = df["review_priority"].value_counts().reindex(["Critical","High","Moderate","Low"], fill_value=0).reset_index()
        counts.columns = ["priority","count"]
        fig = px.pie(counts, names="priority", values="count", hole=.58, color="priority", color_discrete_map={"Critical":"#c2410c","High":"#f59e0b","Moderate":"#eab308","Low":"#22c55e"})
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel"><h3>Planning signals</h3>', unsafe_allow_html=True)
    drivers = df["dominant_driver"].value_counts().reset_index()
    drivers.columns = ["driver","records"]
    fig = px.bar(drivers, x="records", y="driver", orientation="h", color="records", color_continuous_scale="Blues")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Shelter Capacity":
    st.markdown('<div class="panel"><h3>🏠 Temporary shelter capacity</h3><div class="small-note">Compare facility capacity with current bookings and incoming animal demand.</div>', unsafe_allow_html=True)
    view = df[["record_id","zone","site_name","animal_type","animal_count","facility_capacity","current_booked","occupancy_pct","capacity_gap","shelter_pressure_score","review_priority"]].copy()
    view = view.sort_values("shelter_pressure_score", ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    a,b = st.columns(2)
    with a:
        agg = df.groupby("site_name", as_index=False).agg(capacity=("facility_capacity","sum"), booked=("current_booked","sum"))
        agg["open"] = (agg["capacity"]-agg["booked"]).clip(lower=0)
        fig = px.bar(agg, x="site_name", y=["capacity","booked"], barmode="group", labels={"value":"Animal spaces","variable":"Measure"})
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        hist = px.histogram(df, x="occupancy_pct", nbins=10, labels={"occupancy_pct":"Current occupancy (%)"})
        hist.update_layout(height=350, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(hist, use_container_width=True)

elif page == "Evacuation & Transport":
    st.markdown('<div class="panel"><h3>🚚 Evacuation and transport readiness</h3><div class="small-note">Screen animal transport gaps and travel-time pressure before temporary shelter intake.</div>', unsafe_allow_html=True)
    v = df.copy()
    v["transport_coverage_pct"] = np.where(v["animal_count"]>0, 100*v["transport_capacity"]/v["animal_count"],100).clip(0,100)
    v = v[["record_id","zone","site_name","animal_type","animal_count","transport_capacity","transport_coverage_pct","travel_minutes","evacuation_priority","transport_gap","shelter_pressure_score","review_priority"]].sort_values("transport_gap", ascending=False)
    st.dataframe(v, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        z = df.groupby("zone", as_index=False).agg(animals=("animal_count","sum"), transport=("transport_capacity","sum"))
        fig = px.bar(z, x="zone", y=["animals","transport"], barmode="group", labels={"value":"Animals / transport spaces","variable":"Measure"})
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.scatter(df, x="travel_minutes", y="transport_gap", size="animal_count", color="evacuation_priority", hover_name="site_name")
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Animal Mix":
    st.markdown('<div class="panel"><h3>🐕 Animal mix and intake profile</h3><div class="small-note">Understand species mix and where pet/livestock handling capacity may differ.</div>', unsafe_allow_html=True)
    mix = df.groupby(["animal_type"], as_index=False).agg(animals=("animal_count","sum"), records=("record_id","count"), avg_score=("shelter_pressure_score","mean"))
    st.dataframe(mix.sort_values("animals", ascending=False), use_container_width=True, hide_index=True)
    fig = px.treemap(mix, path=["animal_type"], values="animals", color="avg_score", color_continuous_scale="YlOrRd")
    fig.update_layout(height=430, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Facilities & Support":
    st.markdown('<div class="panel"><h3>🛠️ Facilities, supplies and support</h3><div class="small-note">Review water/feed coverage, backup power, staffing, handling space and care support.</div>', unsafe_allow_html=True)
    v = df[["record_id","site_name","animal_type","water_capacity_l_day","feed_capacity_kg_day","supply_days","backup_power_hours","handling_staff","crate_pen_availability","veterinary_support","livestock_support","last_inspection_days","inspection_pressure"]].copy()
    st.dataframe(v.sort_values("supply_days"), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        fig = px.scatter(df, x="supply_days", y="backup_power_hours", size="animal_count", color="review_priority", hover_name="site_name", labels={"supply_days":"Minimum supply runway (days)"})
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        support = pd.DataFrame({"signal":["Handling staff","Crate/pen availability","Veterinary support","Livestock support"], "mean":[df.handling_staff.mean()/12*100, df.crate_pen_availability.mean()/max(df.crate_pen_availability.max(),1)*100, df.veterinary_support.mean()/2*100, df.livestock_support.mean()/2*100]})
        fig = px.bar(support, x="signal", y="mean", range_y=[0,100], labels={"mean":"Relative readiness (%)","signal":"Support signal"})
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Priority Queue":
    st.markdown('<div class="panel"><h3>🚨 Priority review queue</h3><div class="small-note">Planning queue based on capacity, transport, travel, supplies, support and inspection signals.</div>', unsafe_allow_html=True)
    q = df.sort_values(["shelter_pressure_score","animal_count"], ascending=False).copy()
    cols = ["record_id","zone","site_name","animal_type","animal_count","shelter_pressure_score","review_priority","dominant_driver","capacity_gap","transport_gap","travel_minutes","supply_days"]
    st.dataframe(q[cols], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Scenario Lab":
    st.markdown('<div class="panel"><h3>🧪 Scenario Lab</h3><div class="small-note">Explore how planning changes could alter screening pressure. Scenarios are analytical, not operational instructions.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        cap = st.slider("Facility capacity change", -50, 100, 0, 5)
        trans = st.slider("Transport capacity change", -50, 100, 0, 5)
    with c2:
        staff = st.slider("Handling staff change", -6, 12, 0, 1)
        travel = st.slider("Travel-time change (%)", -40.0, 80.0, 0.0, 5.0)
    with c3:
        supply = st.slider("Supply runway change (days)", -3.0, 7.0, 0.0, 0.5)
        insp = st.slider("Inspection age change (days)", -60, 180, 0, 10)
    base = enrich(raw)
    scen = scenario_adjust(raw, cap, trans, staff, travel, supply, insp)
    compare = pd.DataFrame({
        "Metric":["Average score","High/Critical records","Capacity gap","Transport gap","Average supply runway"],
        "Baseline":[base.shelter_pressure_score.mean(), int(base.review_priority.isin(["High","Critical"]).sum()), int(base.capacity_gap.sum()), int(base.transport_gap.sum()), base.supply_days.mean()],
        "Scenario":[scen.shelter_pressure_score.mean(), int(scen.review_priority.isin(["High","Critical"]).sum()), int(scen.capacity_gap.sum()), int(scen.transport_gap.sum()), scen.supply_days.mean()],
    })
    st.dataframe(compare.round(2), use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=base.site_name, y=base.shelter_pressure_score)
    fig.add_bar(name="Scenario", x=scen.site_name, y=scen.shelter_pressure_score)
    fig.update_layout(barmode="group", height=380, margin=dict(l=10,r=10,t=15,b=10), yaxis_title="Pressure score", xaxis_title="Shelter site")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Reports & Export":
    st.markdown('<div class="panel"><h3>📄 Reports & export</h3><div class="small-note">Generate a compact local planning report from the currently loaded CSV.</div>', unsafe_allow_html=True)
    q = df.sort_values("shelter_pressure_score", ascending=False)
    report = f"""# Disaster Animal Shelter Planning Report\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n## Summary\n\n- Shelter sites: {summary['sites']}\n- Zones: {summary['zones']}\n- Animals: {summary['animals']:,}\n- Open capacity: {summary['available_capacity']:,}\n- High/Critical review records: {summary['high_priority']}\n- Average screening pressure: {summary['avg_score']:.1f}/100\n\n## Priority records\n\n{markdown_table(q[["record_id","site_name","animal_type","animal_count","shelter_pressure_score","review_priority","dominant_driver"]], 15)}\n\n## Intended use\n\nScreening outputs organize local operational signals for additional review. They do not guarantee shelter availability, evacuation outcomes, animal welfare outcomes, or emergency-response performance.\n"""
    st.text_area("Report preview", report, height=430)
    st.download_button("Download Markdown report", report.encode("utf-8"), file_name="disaster_animal_shelter_report.md", mime="text/markdown")
    st.download_button("Download enriched CSV", df.to_csv(index=False).encode("utf-8"), file_name="disaster_animal_shelter_enriched.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Explorer":
    st.markdown('<div class="panel"><h3>🔎 Data Explorer</h3><div class="small-note">Inspect the raw input and derived fields used throughout the dashboard.</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Synthetic demonstration data • Local CSV processing • Screening and planning support only")
