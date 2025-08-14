from __future__ import annotations
import time, threading, queue, random
from dataclasses import dataclass
from typing import Optional
@dataclass
class SensorEvent:
    timestamp: float; sensor_id: str; metric: str; value: float; unit: str
METRICS=[("temperature","C",23.5,1.0),("vibration","mm_s",3.0,1.0),("power","kW",4.2,0.8),("flow","m3h",100.0,10.0)]
class SensorSimulator:
    def __init__(self, hz: float=1.0, anomaly_rate: float=0.02):
        self.hz=hz; self.anomaly_rate=anomaly_rate; self.q=queue.Queue(); self._stop=False; self.thread=None
    def start(self):
        self._stop=False; self.thread=threading.Thread(target=self._run, daemon=True); self.thread.start()
    def stop(self):
        self._stop=True
        if self.thread: self.thread.join(1.0)
    def _run(self):
        while not self._stop:
            for name,unit,mean,std in METRICS:
                val=random.gauss(mean,std)
                if random.random()<self.anomaly_rate:
                    if name=="vibration": val+=random.uniform(5,9)
                    elif name=="temperature": val+=random.uniform(3,7)
                    elif name=="power": val+=random.uniform(2,4)
                    elif name=="flow": val-=random.uniform(30,50)
                self.q.put(SensorEvent(time.time(),"PUMP3",name,float(val),unit))
            time.sleep(1.0/max(self.hz,0.1))
    def read(self, timeout: float=0.1)->Optional[SensorEvent]:
        try: return self.q.get(timeout=timeout)
        except queue.Empty: return None
