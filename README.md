# Customer Segmentation App

A Streamlit app with two views:
- **Segment Report** — cluster profiles, insights, and interactive Plotly charts
- **Predict Your Segment** — enter RFM values and get a live cluster prediction

## 1. Export the model from your notebook

Run this in Colab **after** fitting your final K-Means model (k=3) and scaler:

```python
import pickle

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

from google.colab import files
files.download('kmeans_model.pkl')
files.download('scaler.pkl')
```

This downloads two files — put them in the same folder as `app.py`.

## 2. Project structure

```
your-repo/
├── app.py
├── requirements.txt
├── kmeans_model.pkl
├── scaler.pkl
└── README.md
```

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "Customer segmentation app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 5. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **New app**, select your repo, branch `main`, and file `app.py`
3. Click **Deploy** — you'll get a shareable public link

## Notes

- Cluster labels ("VIP", "Core", "At-Risk") are assigned dynamically based on
  the trained centroids, so this still works even if K-Means assigns different
  raw cluster numbers (0/1/2) on a re-run.
- If you retrain the model with different features or a different k, update
  `PROFILES` in `app.py` accordingly (currently written for 3 clusters).
