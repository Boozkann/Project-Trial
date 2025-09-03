# app.py
import json,pandas as pd,streamlit as st,matplotlib.pyplot as plt
from fetch_wb import fetch_wb_panel
from features import build_features
from model_unsupervised import fit_iforest,score_risk,RISK_FEATURES_DEFAULT
from rating import simple_inverse,percentile_inverse,score_to_band

st.set_page_config(page_title="Sovereign Risk v2",layout="wide")
st.title("Sovereign Credit Risk — PoC v2")

with open("indicator_map.json") as f: IND_MAP=json.load(f)
DEFAULT=["DEU","GRC","ARG","ESP","ITA"]
countries=st.sidebar.multiselect("Countries",DEFAULT,default=DEFAULT)
contamination=st.sidebar.slider("Contamination",0.01,0.4,0.15,0.01)
risk_threshold=st.sidebar.slider("Risk threshold",0.5,0.95,0.65,0.01)
scaler=st.sidebar.selectbox("Scaler",["simple_inverse","percentile_inverse"])
run=st.sidebar.button("Run")

disp={"current_account_gdp":"Current Account / GDP (%)",
"central_govt_debt_gdp":"Central Govt Debt / GDP (%)",
"cash_surplus_deficit_gdp":"Cash Surplus/Deficit / GDP (%)",
"debt_service_to_exports":"Debt Service / Exports (%)",
"reserves_months_imports":"Reserves (months imports)",
"inflation_cpi":"Inflation CPI (%)",
"gdp_growth":"GDP Growth (%)",
"unemployment_rate":"Unemployment (%)",
"trade_openness":"Trade Openness (% GDP)",
"political_stability_estimate":"Political Stability (WGI)"}

def plot_series(df, col, title):
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(4, 2.3))
    ax.plot(df.index, df[col])
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)
    plt.close(fig)   # 🧹 grafiği kapat, bellek temizlensin


def process(c):
    raw=fetch_wb_panel(c,IND_MAP)
    if raw.empty: return None,None,None
    feats=build_features(raw)
    use=[k for k in RISK_FEATURES_DEFAULT if k in feats.columns]
    X=feats[use].fillna(feats[use].median(numeric_only=True))
    m=fit_iforest(X,contamination)
    s=score_risk(m,X)
    return raw,feats,s

if run:
    results={}; allrp=[]
    for c in countries:
        raw,feats,s=process(c)
        if raw is None: continue
        results[c]={"raw":raw,"scores":s}; allrp.append(s["risk_prob"])
    ref=pd.concat(allrp) if allrp else pd.Series(dtype=float)
    cols=st.columns(len(results))
    for i,(c,v) in enumerate(results.items()):
        s=v["scores"]; y=s.index.max(); rp=float(s.loc[y,"risk_prob"])
        if scaler=="percentile_inverse" and not ref.empty:
            rating=float(percentile_inverse(pd.Series([rp]),ref).iloc[0])
        else: rating=float(simple_inverse(pd.Series([rp])).iloc[0])
        band=score_to_band(rating)
        flag="AT RISK" if rp>=risk_threshold else "LOW/MOD"
        with cols[i]: st.metric(c,f"Rating {rating:.2f}",f"Risk {rp:.3f}")
        st.caption(f"{c} {y} → {band}, Flag={flag}")
    st.subheader("Risk Probability over Time")
    fig,ax=plt.subplots(figsize=(9,3))
    for c,v in results.items():
        ax.plot(v["scores"].index,v["scores"]["risk_prob"],label=c)
    ax.legend(); st.pyplot(fig)
    st.subheader("Indicators")
    for c,v in results.items():
        st.markdown(f"### {c}")
        raw=v["raw"]
        for col in disp:
            if col in raw.columns: plot_series(raw,col,disp[col])
