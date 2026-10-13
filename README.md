# Catching more defaulters (leak-free stacked ensemble)

Streamlit app for the poster *An Adaptive Stacking Framework with Feature Selection and Imbalance Handling for Credit Risk Prediction* (Maoundoe Lucresse, M. M. Lawal, A. F. D. Kana).

Imputing, scaling, Information-Gain feature selection and class weighting all live inside one scikit-learn Pipeline, so they are fitted on training data only (no leakage).

## Fastest free way online: Streamlit Community Cloud (no Python on your computer)
1. Create a free account at github.com. Click **+ > New repository**, name it `credit-risk-demo`, set it **Public**, and click **Create repository**.
2. Click **uploading an existing file**, drag in `app.py`, `train.py`, `requirements.txt` and `README.md`, then **Commit changes**.
3. Go to share.streamlit.io, click **Continue with GitHub** and authorise Streamlit.
4. Click **Create app** > **Deploy a public app from GitHub**. Repository: `credit-risk-demo`, Branch: `main`, Main file path: `app.py`. Pick an app URL (for example `credit-risk-abu`) and click **Deploy**.
5. After 3 to 5 minutes your app opens at `https://<your-url>.streamlit.app`. Open the **Train on a CSV** tab, upload a dataset, choose the target column and which value means default, and press **Train**.
6. Turn the app link into a QR code for the poster.

Free apps go to sleep when unused (click "wake up" on the page) and forget models trained in the app after a restart, so open the link a few minutes before presenting and train again (about a minute) if needed. Use Fast mode for large files. Hugging Face Spaces now needs a paid plan for Streamlit/Docker apps, so this guide uses Streamlit Community Cloud instead.

## Option B: train on your computer and upload the models
```
pip install -r requirements.txt
python train.py --csv german.csv     --target class   --name German     --k 13
python train.py --csv australian.csv --target class   --name Australian --k 9  --positive 0
python train.py --csv taiwan.csv     --target default --name Taiwan     --k 17 --folds 5
streamlit run app.py        # test locally at http://localhost:8501
```
Upload the resulting `models/` folder to the Space too. Pin your scikit-learn version in `requirements.txt` (for example `scikit-learn==1.5.2`, check with `pip show scikit-learn`). The target must mean 1 = defaulter (use `--positive` to map another label).

## Notes
- Research prototype, not for real lending decisions.
- Metrics from the in-app trainer are on a 25% hold-out (fast mode uses at most 10,000 rows). `train.py` reports cross-validated metrics.
