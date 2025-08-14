import streamlit as st, pandas as pd
from src.sensor_stream import SensorSimulator
from src.predict import FailurePredictor
from src.anomaly import AnomalyDetector
from src.rag_pipeline import RAGEngine

st.set_page_config(page_title="IoT Sensor Data RAG for Smart Buildings", page_icon="🏢", layout="wide")
st.title("🏢 IoT Sensor Data RAG for Smart Buildings")
st.caption("Predictive maintenance • Anomaly detection • RAG over manuals & specs")

st.sidebar.header("Configuration")
manuals_dir = st.sidebar.text_input("Manuals directory", "data/manuals")
specs_dir = st.sidebar.text_input("Specs directory", "data/specs")
persist_dir = st.sidebar.text_input("Vector DB directory", "chroma_db")
col_a, col_b = st.sidebar.columns(2)
with col_a: hz = st.number_input("Sensor Hz", 0.1, 10.0, 1.0, 0.1)
with col_b: anomaly_rate = st.number_input("Anomaly rate", 0.0, 0.5, 0.05, 0.01)

tab1, tab2, tab3 = st.tabs(["📡 Live Sensors", "🧠 RAG Assistant", "📈 Models & Evaluation"])

if "sim" not in st.session_state:
    st.session_state.sim = SensorSimulator(hz=hz, anomaly_rate=anomaly_rate); st.session_state.sim.start()
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["timestamp","sensor_id","metric","value","unit"])
if "fp" not in st.session_state:
    st.session_state.fp = FailurePredictor.train_synthetic()
if "ad" not in st.session_state:
    base = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="min"),
        "sensor_id": ["PUMP3"]*200,
        "metric": ["temperature"]*50 + ["vibration"]*50 + ["power"]*50 + ["flow"]*50,
        "value": ([24]*50 + [3]*50 + [4]*50 + [100]*50),
        "unit": ["C"]*50 + ["mm_s"]*50 + ["kW"]*50 + ["m3h"]*50,
    })
    st.session_state.ad = AnomalyDetector.train_from_baseline(base)
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine.initialize(manuals_dir, specs_dir, persist_dir)

with tab1:
    left, right = st.columns([3,2], gap="large")
    with left:
        st.subheader("Live stream"); ph = st.empty(); alerts = st.container()
        run = st.checkbox("Run stream", value=True); max_rows = st.number_input("Max rows to keep", 100, 5000, 500, 50)
        while run and st.runtime.exists():
            ev = st.session_state.sim.read(timeout=0.3)
            if ev is None: st.pause(0.05); continue
            row = {"timestamp": pd.to_datetime(ev.timestamp, unit="s"), "sensor_id": ev.sensor_id, "metric": ev.metric, "value": ev.value, "unit": ev.unit}
            df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True).tail(int(max_rows))
            st.session_state.df = df; ph.dataframe(df, use_container_width=True, hide_index=True)
            vib = float(df[df.metric=="vibration"].value.tail(1).mean()) if not df[df.metric=="vibration"].empty else 3.0
            temp = float(df[df.metric=="temperature"].value.tail(1).mean()) if not df[df.metric=="temperature"].empty else 24.0
            power = float(df[df.metric=="power"].value.tail(1).mean()) if not df[df.metric=="power"].empty else 4.0
            flow = float(df[df.metric=="flow"].value.tail(1).mean()) if not df[df.metric=="flow"].empty else 100.0
            p_fail = st.session_state.fp.predict_proba({"temperature":temp,"vibration":vib,"power":power,"flow":flow})
            flags=[]
            if vib>7.1: flags.append(f"Vibration high: {vib:.2f} mm/s")
            if temp>28.0: flags.append(f"Temperature high: {temp:.2f} °C")
            if power>6.0: flags.append(f"Power high: {power:.2f} kW")
            if flow<80.0: flags.append(f"Flow low: {flow:.2f} m³/h")
            if flags or p_fail>0.5:
                with alerts:
                    st.error(" | ".join(flags) if flags else "Potential failure risk detected.")
                    st.write(f"Predicted failure risk: **{p_fail:.2f}**")
            st.pause(0.25)
    with right:
        st.subheader("Current status")
        df = st.session_state.df
        if not df.empty:
            last = df.groupby("metric")["value"].last().to_dict()
            st.metric("Temperature (°C)", f"{last.get('temperature',24.0):.2f}")
            st.metric("Vibration (mm/s)", f"{last.get('vibration',3.0):.2f}")
            st.metric("Power (kW)", f"{last.get('power',4.0):.2f}")
            st.metric("Flow (m³/h)", f"{last.get('flow',100.0):.2f}")
        st.divider()
        if not df.empty:
            st.line_chart(df[df.metric=="temperature"][["timestamp","value"]].set_index("timestamp"))
            st.line_chart(df[df.metric=="vibration"][["timestamp","value"]].set_index("timestamp"))
            st.line_chart(df[df.metric=="power"][["timestamp","value"]].set_index("timestamp"))
            st.line_chart(df[df.metric=="flow"][["timestamp","value"]].set_index("timestamp"))

with tab2:
    st.subheader("Ask the manuals/specs")
    query = st.text_input("Query", "Pump 3 shows high vibration and low flow. What should I check?")
    if st.button("Retrieve & Answer"):
        result = st.session_state.rag.ask(query, k=4)
        st.markdown("### Answer"); st.write(result["answer"])
        with st.expander("Retrieved context"):
            for ch in result["chunks"]:
                st.markdown(f"**Source:** {ch.get('source','unknown')}"); st.text(ch.get("page_content",""))

with tab3:
    st.subheader("Failure Predictor")
    temp = st.slider("Temperature (°C)", 18.0, 40.0, 24.0, 0.1)
    vib = st.slider("Vibration (mm/s)", 0.0, 15.0, 3.0, 0.1)
    power = st.slider("Power (kW)", 0.0, 12.0, 4.0, 0.1)
    flow = st.slider("Flow (m³/h)", 0.0, 200.0, 100.0, 1.0)
    p = st.session_state.fp.predict_proba({"temperature":temp,"vibration":vib,"power":power,"flow":flow})
    st.metric("Predicted failure risk", f"{p:.2f}")
    st.write(st.session_state.fp.recommendation({"temperature":temp,"vibration":vib,"power":power,"flow":flow}))
    st.divider()
    st.subheader("Basic RAG Metrics (Mock)")
    st.write("• Retrieval@4 hit-rate (manual sanity check): ~0.9")
    st.write("• Average latency (local embedding + Chroma): ~200–600 ms on typical CPU")
    st.write("• To run RAGAS, add real Q/A pairs and compute factual consistency (see README).")
st.sidebar.divider(); st.sidebar.info("Tip: Add your own PDFs/text under data/ and restart to re-index.")
