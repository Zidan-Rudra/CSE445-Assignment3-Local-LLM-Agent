"""
CSE445 Assignment #3 - Machine Learning tools
North South University | CSE445 Machine Learning
"""
import json, math
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}

def _load(name):
    name = str(name).lower().strip()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")
    return DATASETS[name]()

def load_dataset_summary(dataset_name: str) -> str:
    data = _load(dataset_name)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return json.dumps({
        "dataset": dataset_name.lower().strip(),
        "n_samples": int(df.shape[0]),
        "n_features": int(len(data.feature_names)),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum())
    })

def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    data = _load(dataset_name)
    if not 0.1 <= float(test_size) <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=float(test_size),
        random_state=42, stratify=data.target
    )
    mt = str(model_type).lower().strip()
    if mt == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif mt == "logistic_regression":
        clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    elif mt == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif mt == "svc":
        clf = Pipeline([("scale", StandardScaler()), ("svc", SVC(C=1.0, kernel="rbf"))])
    else:
        raise ValueError(f"Unsupported model '{mt}'")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cv = cross_val_score(clf, data.data, data.target, cv=5)
    return json.dumps({"model": mt, "dataset": dataset_name.lower().strip(),
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "cv_mean_accuracy": round(float(cv.mean()), 4),
        "cv_std": round(float(cv.std()), 4)})

def hyperparameter_tuning(dataset_name: str, model_type: str = "svc") -> str:
    data = _load(dataset_name)
    mt = str(model_type).lower().strip()
    if mt == "svc":
        pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
        grid = {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}
    elif mt == "decision_tree":
        pipe = DecisionTreeClassifier(random_state=42)
        grid = {"max_depth": [2, 3, 4, 6, None], "min_samples_split": [2, 5]}
    else:
        raise ValueError("model_type must be 'svc' or 'decision_tree'")
    search = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(data.data, data.target)
    return json.dumps({"dataset": dataset_name, "model": mt,
        "best_params": search.best_params_, "best_cv_accuracy": round(float(search.best_score_), 4)})

def feature_reduction(dataset_name: str, n_components: int = 2) -> str:
    data = _load(dataset_name)
    n_components = int(n_components)
    if not 1 <= n_components < data.data.shape[1]:
        raise ValueError("n_components must be smaller than the feature count")
    pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components))])
    Xt = pipe.fit_transform(data.data)
    explained = pipe.named_steps["pca"].explained_variance_ratio_
    return json.dumps({"dataset": dataset_name, "original_features": int(data.data.shape[1]),
        "reduced_features": n_components,
        "explained_variance_ratio": [round(float(x), 4) for x in explained],
        "total_explained_variance": round(float(explained.sum()), 4)})

class RegularizedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x): return self.net(x)

def train_pytorch_regularized(dataset_name: str, hidden_dim: int = 32, epochs: int = 50,
                              lr: float = 0.01, dropout: float = 0.25) -> str:
    data = _load(dataset_name)
    torch.manual_seed(42); np.random.seed(42)
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2,
                                          random_state=42, stratify=data.target)
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-7
    Xtr, Xte = (Xtr-mean)/std, (Xte-mean)/std
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    Xv, yv = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long)
    model = RegularizedMLP(Xt.shape[1], int(hidden_dim), len(np.unique(ytr)), float(dropout))
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(lr))
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    for _ in range(int(epochs)):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        if not torch.isfinite(loss): raise FloatingPointError("NaN/Inf loss detected")
        loss.backward(); opt.step(); sched.step(loss)
    model.eval()
    with torch.no_grad():
        acc = (model(Xv).argmax(1) == yv).float().mean().item()
    return json.dumps({"framework":"PyTorch", "dataset":dataset_name, "hidden_dim":int(hidden_dim),
        "epochs":int(epochs), "dropout":float(dropout), "final_loss":round(float(loss.item()),4),
        "test_accuracy":round(float(acc),4)})

AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "hyperparameter_tuning": hyperparameter_tuning,
    "feature_reduction": feature_reduction,
    "train_pytorch_regularized": train_pytorch_regularized,
}
