# Disaster Animal Shelter Planner

A 100% local Python + Streamlit planning dashboard for temporary animal shelter capacity during disasters. It uses locally supplied evacuation exposure, animal counts, transport, facility capacity, supplies, staffing, and support readiness signals.

## Highlights
- Explainable 0–100 shelter-pressure screening score
- Low / Moderate / High / Critical planning classification
- Capacity and occupancy monitoring
- Transport coverage and travel-time analysis
- Pet vs livestock intake profiling
- Water/feed runway and backup-power monitoring
- Handling staff, crate/pen, veterinary and livestock support assessment
- Priority review queue
- What-if scenario analysis
- Local Markdown and CSV exports
- Synthetic sample dataset and blank template
- No external APIs or cloud services required

## Run locally (from the project root)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest tests/ -q
streamlit run app.py
```

## Data
Use `data/sample_disaster_animal_shelter.csv` for a ready-to-run demo or `data/disaster_animal_shelter_template.csv` for your own local records.

## Responsible use
Outputs are screening and planning signals. They do not guarantee shelter availability, evacuation outcomes, animal welfare outcomes, or emergency-response performance, and they should not replace emergency management protocols or qualified operational judgment.


The bundled ZIP is root-level: `app.py`, `requirements.txt`, `data/`, `assets/`, `tests/`, and `doc/` are directly at the project root.
