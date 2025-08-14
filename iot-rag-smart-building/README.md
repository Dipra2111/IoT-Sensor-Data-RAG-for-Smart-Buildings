# IoT Sensor Data RAG for Smart Buildings 🏢
Predictive maintenance + anomaly detection + retrieval-augmented guidance over manuals/specs.

## Features
- Live sensor stream (simulator; pluggable to MQTT later)
- RAG over manuals & building specs (all-MiniLM-L6-v2 + Chroma)
- Predictive failure risk (RandomForest)
- Anomaly alerts (rule-based; extendable to IsolationForest)
- Streamlit UI (Sensors, RAG, Models)

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```
Drop your PDFs/txt into `data/manuals` and `data/specs`, then restart to re-index.

## Deploy on Hugging Face Spaces
- Create Space → SDK: Streamlit → Public
- Upload files or push via git
- In **Settings → Build → Pre-build commands** add:
```bash
python download_model.py
```

## Docs
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
