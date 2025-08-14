from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
@dataclass
class FailurePredictor:
    model: RandomForestClassifier
    @classmethod
    def train_synthetic(cls, n: int = 2000) -> "FailurePredictor":
        rng = np.random.default_rng(42)
        X = rng.normal(0,1,(n,4))
        y = ((X[:,1]>1.0)&(X[:,0]>0.8)) | ((X[:,2]>1.2)&(X[:,3]<-0.5))
        y = y.astype(int)
        X[:,0]=X[:,0]*2+24; X[:,1]=X[:,1]*2+3; X[:,2]=X[:,2]*1.5+4; X[:,3]=X[:,3]*10+100
        model = RandomForestClassifier(n_estimators=200, random_state=42); model.fit(X,y); return cls(model)
    def predict_proba(self, features: dict) -> float:
        import numpy as np
        x = np.array([[features.get("temperature",24.0),features.get("vibration",3.0),features.get("power",4.0),features.get("flow",100.0)]])
        return float(self.model.predict_proba(x)[0,1])
    def recommendation(self, f: dict) -> str:
        temp=f.get("temperature",24.0); vib=f.get("vibration",3.0); power=f.get("power",4.0); flow=f.get("flow",100.0)
        tips=[]
        if vib>7.1: tips.append("Vibration exceeds ISO 10816 alarm. Inspect bearing & alignment; check cavitation.")
        if temp>28.0: tips.append("Temperature elevated. Check cooling water and load conditions.")
        if power>6.0: tips.append("Power above baseline. Verify filters, belts, and damper positions.")
        if flow<80.0: tips.append("Low flow. Inspect strainer and valve positions.")
        return " ".join(tips) if tips else "System within normal parameters. Continue routine checks."
