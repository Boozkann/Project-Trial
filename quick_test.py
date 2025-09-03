# quick_test.py
import json,sys,pandas as pd
from fetch_wb import fetch_wb_panel
from features import build_features
from model_unsupervised import fit_iforest,score_risk,RISK_FEATURES_DEFAULT
from rating import simple_inverse,score_to_band

COUNTRY=sys.argv[1] if len(sys.argv)>1 else "DEU"
with open("indicator_map.json") as f: ind_map=json.load(f)
raw=fetch_wb_panel(COUNTRY,ind_map)
feats=build_features(raw)
use_cols = [c for c in RISK_FEATURES_DEFAULT if c in feats.columns]

# 1) Tümü NaN kolonları at
X = feats[use_cols].dropna(axis=1, how="all")

# 2) Kolon medyanlarını al; medyanı NaN çıkan kolonları (tamamen boş demektir) at
col_median = X.median(numeric_only=True)
good_cols = [c for c in X.columns if pd.notna(col_median.get(c))]
X = X[good_cols]

# 3) Kalan boşlukları medyanla doldur
X = X.fillna(X.median(numeric_only=True))

# 4) Hâlâ NaN’li satır kalırsa at (çok nadir)
X = X.dropna(axis=0, how="any")

# 5) Çok az özellik kalırsa uyar (model 3–4 özellikten azsa zayıf olur)
if X.shape[1] < 5:
    print(f"[WARN] Only {X.shape[1]} usable features. Results may be unstable.")
model=fit_iforest(X,contamination=0.10)
scores=score_risk(model,X)
latest_year=scores.index.max()
rp=float(scores.loc[latest_year,"risk_prob"])
rating=float(simple_inverse(pd.Series([rp])).iloc[0])
band=score_to_band(rating)
print(f"Year {latest_year} {COUNTRY}: risk_prob={rp:.3f}, rating={rating:.2f}, band={band}")
