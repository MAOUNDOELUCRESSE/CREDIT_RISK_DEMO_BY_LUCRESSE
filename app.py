import json
from pathlib import Path
import joblib, pandas as pd, streamlit as st
from train import quick_train

st.set_page_config(page_title="Credit risk: catching more defaulters", page_icon="🏦", layout="wide")
MODELS = sorted(p.stem for p in Path("models").glob("*.json"))

st.title("CREDIT RISK PREDICTION")
st.write("A leak-free stacked ensemble with class weighting. Pick a dataset, review how it performed, then score a borrower.")
have = bool(MODELS)
if have:
    name = st.sidebar.selectbox("Dataset", MODELS, format_func=lambda s: s.replace("_", " ").title())
    meta = json.load(open(f"models/{name}.json"))
    model = joblib.load(f"models/{name}.joblib")
else:
    st.info("No model yet. Open the **Train on a CSV** tab, upload a dataset and press Train. It takes about a minute.")

tab0, tab1, tab2, tab3 = st.tabs(["Train on a CSV", "Performance", "Score a borrower", "Score a CSV"])

with tab0:
    st.write("Upload a credit dataset (CSV). The app trains the leak-free pipeline with and without class weighting and keeps the weighted model.")
    up = st.file_uploader("Dataset (CSV)", type="csv", key="train_csv")
    if up:
        df = pd.read_csv(up)
        st.dataframe(df.head(5))
        target = st.selectbox("Target column (the outcome)", df.columns, index=len(df.columns) - 1)
        label = st.selectbox("Which value means DEFAULT?", sorted(df[target].dropna().astype(str).unique()))
        ignore = st.multiselect("Columns to ignore (for example an ID)", [c for c in df.columns if c != target])
        c1, c2, c3 = st.columns(3)
        nm = c1.text_input("Name for this model", Path(up.name).stem.title())
        k = c2.number_input("Features to keep (Information Gain)", 2, 60, 13)
        fast = c3.checkbox("Fast mode (recommended for big files)", True)
        if st.button("Train", type="primary"):
            with st.spinner("Training... about a minute for small files"):
                quick_train(df, target, nm, int(k), positive=label, ignore=ignore, fast=fast)
            st.success("Done. Open the Performance and Score tabs.")
            st.rerun()

with tab1:
    if not have:
        st.info("Train a model first (first tab).")
    else:
        m = meta["metrics"]
        st.caption(f"{meta['rows']:,} borrowers, {meta['default_share']}% defaulters. Evaluated on held-out borrowers; every step fitted on training data only.")
        c = st.columns(4)
        for col, (lab, key) in zip(c, [("Defaulters caught (recall, %)", "recall"), ("Precision (%)", "precision"), ("F1 (%)", "f1"), ("MCC", "mcc")]):
            col.metric(lab, m["class_weighted"][key], round(m["class_weighted"][key] - m["standard"][key], 2))
        st.write("Change shown is class-weighted versus the standard model.")
        st.dataframe(pd.DataFrame(m).T.rename(index={"standard": "Standard", "class_weighted": "Class-weighted"}))
        st.write(f"Out of 100 real defaulters, the class-weighted model catches about **{round(m['class_weighted']['recall'])}**; the standard model about **{round(m['standard']['recall'])}**.")

with tab2:
    if not have:
        st.info("Train a model first (first tab).")
    else:
        st.write("Enter the borrower's details. Defaults are typical values from the training data.")
        row, cols = {}, st.columns(3)
        for i, (f, info) in enumerate(meta["features"].items()):
            with cols[i % 3]:
                if info["type"] == "num":
                    lo, hi = info["min"], info["max"]
                    row[f] = st.number_input(f, value=float(info["median"]), min_value=lo, max_value=hi if hi > lo else lo + 1.0)
                else:
                    row[f] = st.selectbox(f, info["values"])
        thr = st.slider("Decision threshold for 'high risk'", 0.05, 0.95, 0.5, 0.05)
        if st.button("Score borrower", type="primary"):
            p = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
            st.progress(min(max(p, 0.0), 1.0))
            st.metric("Estimated probability of default", f"{p:.0%}")
            (st.error if p >= thr else st.success)("High risk: review before lending" if p >= thr else "Lower risk at this threshold")

with tab3:
    if not have:
        st.info("Train a model first (first tab).")
    else:
        up2 = st.file_uploader("CSV with the same columns (without the target)", type="csv", key="score_csv")
        if up2:
            d = pd.read_csv(up2)
            missing = [f for f in meta["features"] if f not in d.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                d["default_probability"] = model.predict_proba(d[list(meta["features"])])[:, 1].round(3)
                st.dataframe(d)
                st.download_button("Download scored CSV", d.to_csv(index=False), "scored.csv")
st.caption("Research prototype from the 3rd Biennial Postgraduate Conference 2026 poster (Maoundoe Lucresse, M. M. Lawal, A. F. D. Kana). Not for real lending decisions.")
