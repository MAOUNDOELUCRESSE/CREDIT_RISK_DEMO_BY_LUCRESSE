"""Leak-free credit-risk training script.

Every step (imputing, scaling, Information-Gain feature selection, class weighting)
lives INSIDE one sklearn Pipeline, so cross-validation fits it on training folds only.
Usage:
  python train.py --csv german.csv --target class --name German --k 13
  (the target column must be 0/1 where 1 = defaulter; use --positive to map a label)
"""
import argparse, json
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build(X, k, weighted, spw=1.0, n_est=300):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    n_feat = len(num) + sum(X[c].nunique() for c in cat)
    cw = "balanced" if weighted else None
    base = [("rf", RandomForestClassifier(n_est, class_weight=cw, random_state=0, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(random_state=0))]
    if XGBClassifier is not None:
        base.append(("xgb", XGBClassifier(n_estimators=n_est, learning_rate=0.1, eval_metric="logloss",
                                          scale_pos_weight=spw if weighted else 1.0, random_state=0, n_jobs=-1)))
    stack = StackingClassifier(base, LogisticRegression(class_weight=cw, max_iter=1000), cv=5, n_jobs=1)
    return Pipeline([("pre", pre), ("fs", SelectKBest(mutual_info_classif, k=min(k, n_feat))), ("stack", stack)])


def quick_train(df, target, name, k=13, positive=None, ignore=(), fast=True):
    """Train standard vs class-weighted on an uploaded DataFrame (25% hold-out) and save the weighted model."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    df = df.drop(columns=list(ignore)).dropna(subset=[target]).reset_index(drop=True)
    y = (df[target].astype(str) == str(positive)).astype(int) if positive is not None else df[target].astype(int)
    X = df.drop(columns=[target])
    if fast and len(X) > 10000:
        keep = X.sample(10000, random_state=0).index
        X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    n_est = 100 if fast else 300
    out, final = {}, None
    for label, weighted in [("standard", False), ("class_weighted", True)]:
        m = build(Xtr, k, weighted, spw, n_est).fit(Xtr, ytr)
        p, pr = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        out[label] = {"recall": round(recall_score(yte, p) * 100, 2), "precision": round(precision_score(yte, p, zero_division=0) * 100, 2),
                      "f1": round(f1_score(yte, p) * 100, 2), "roc_auc": round(roc_auc_score(yte, pr) * 100, 2),
                      "mcc": round(float(matthews_corrcoef(yte, p)), 3)}
        final = m if weighted else final
    Path("models").mkdir(exist_ok=True)
    slug = name.lower().replace(" ", "_")
    joblib.dump(final, f"models/{slug}.joblib")
    num = X.select_dtypes("number").columns.tolist()
    meta = {"name": name, "default_share": round(float(y.mean()) * 100, 1), "rows": int(len(X)),
            "features": {c: ({"type": "num", "min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
                             if c in num else {"type": "cat", "values": sorted(X[c].dropna().astype(str).unique().tolist())}) for c in X.columns},
            "metrics": out}
    json.dump(meta, open(f"models/{slug}.json", "w"), indent=1)
    return meta


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True); a.add_argument("--target", required=True)
    a.add_argument("--name", required=True); a.add_argument("--k", type=int, default=13)
    a.add_argument("--positive", default=None, help="label value meaning 'defaulter' (default: already 1)")
    a.add_argument("--folds", type=int, default=5); a.add_argument("--repeats", type=int, default=1)
    a.add_argument("--sep", default=",")
    a = a.parse_args()

    df = pd.read_csv(a.csv, sep=a.sep).dropna(subset=[a.target])
    y = (df[a.target].astype(str) == str(a.positive)).astype(int) if a.positive is not None else df[a.target].astype(int)
    X = df.drop(columns=[a.target])
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc",
               "mcc": make_scorer(matthews_corrcoef)}
    cv = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=0)
    out = {}
    for label, weighted in [("standard", False), ("class_weighted", True)]:
        r = cross_validate(build(X, a.k, weighted, spw), X, y, cv=cv, scoring=scoring, n_jobs=1)
        out[label] = {m: round(float(np.mean(r["test_" + m])) * 100, 2) if m != "mcc" else round(float(np.mean(r["test_mcc"])), 3) for m in scoring}
        print(label, out[label])

    model = build(X, a.k, True, spw).fit(X, y)
    Path("models").mkdir(exist_ok=True)
    slug = a.name.lower()
    joblib.dump(model, f"models/{slug}.joblib")
    num = X.select_dtypes("number").columns.tolist()
    meta = {"name": a.name, "default_share": round(float(y.mean()) * 100, 1), "rows": int(len(df)),
            "features": {c: ({"type": "num", "min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
                             if c in num else {"type": "cat", "values": sorted(X[c].dropna().astype(str).unique().tolist())})
                         for c in X.columns},
            "metrics": out}
    json.dump(meta, open(f"models/{slug}.json", "w"), indent=1)
    print("saved models/%s.joblib and models/%s.json" % (slug, slug))


if __name__ == "__main__":
    main()
