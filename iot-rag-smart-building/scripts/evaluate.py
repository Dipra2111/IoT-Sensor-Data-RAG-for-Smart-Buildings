import time
from statistics import mean
from src.ingest import CorpusPaths, build_or_load_vectorstore
from src.retriever import retrieve
SAMPLE_QA=[
 {"q":"Pump 3 has high vibration, what checks does the manual suggest?","must_have":"bearing lubrication"},
 {"q":"What is the vibration alarm threshold in the building spec?","must_have":"7.1 mm/s"},
 {"q":"What is the AHU baseline power consumption?","must_have":"3.5–5.0 kW"},
]
def main():
    paths = CorpusPaths("data/manuals","data/specs","chroma_db")
    vs = build_or_load_vectorstore(paths)
    lat, hits = [], 0
    for qa in SAMPLE_QA:
        t0=time.time(); chunks=retrieve(vs, qa["q"], k=4); lat.append(time.time()-t0)
        combined=" ".join([c.get("page_content","") for c in chunks]).lower()
        if qa["must_have"].lower() in combined: hits+=1
    print(f"Queries: {len(SAMPLE_QA)}")
    print(f"Retrieval@4 hit rate: {hits/len(SAMPLE_QA):.2f}")
    print(f"Avg latency (s): {mean(lat):.3f}")
if __name__=="__main__": main()
