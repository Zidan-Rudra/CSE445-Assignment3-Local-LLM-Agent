"""CSE445 benchmark: 3 classical algorithms across 2 datasets with 5-fold CV."""
import json, time
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA = {"iris": load_iris(), "wine": load_wine()}
MODELS = {
    "SVC": Pipeline([("scale", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf"))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows=[]
for dname, d in DATA.items():
    for mname, model in MODELS.items():
        t=time.perf_counter()
        scores=cross_val_score(model,d.data,d.target,cv=cv,scoring="accuracy")
        rows.append({"dataset":dname,"algorithm":mname,"cv_mean":round(float(scores.mean()),4),
                     "cv_std":round(float(scores.std()),4),"min":round(float(scores.min()),4),
                     "max":round(float(scores.max()),4),"elapsed_s":round(time.perf_counter()-t,3)})
print(json.dumps(rows, indent=2))
with open("benchmark_results.json","w") as f: json.dump(rows,f,indent=2)
