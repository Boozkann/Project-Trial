# fetch_wb.py
import requests, pandas as pd, time

WB_BASE = "https://api.worldbank.org/v2"

def _fetch_indicator_series(country_code, indicator, date="2000:2024", per_page=20000):
    page=1; frames=[]
    while True:
        url=f"{WB_BASE}/country/{country_code}/indicator/{indicator}?format=json&date={date}&per_page={per_page}&page={page}"
        r=requests.get(url,timeout=60); r.raise_for_status()
        data=r.json()
        if not isinstance(data,list) or len(data)<2: break
        meta,rows=data
        if not rows: break
        df=pd.DataFrame(rows)
        if df.empty: break
        df=df[['date','value']]; df['date']=pd.to_numeric(df['date'],errors='coerce')
        df=df.dropna(subset=['date']); df['date']=df['date'].astype(int)
        df['value']=pd.to_numeric(df['value'],errors='coerce')
        frames.append(df.set_index('date'))
        if page>=int(meta.get("pages",1)): break
        page+=1; time.sleep(0.1)
    if not frames: return pd.Series(dtype=float)
    return pd.concat(frames)['value']

def fetch_wb_panel(country_code, indicator_map, date="2000:2024"):
    frames=[]
    for code,friendly in indicator_map.items():
        s=_fetch_indicator_series(country_code, code, date)
        if s.empty: continue
        s=s.rename(friendly)
        frames.append(s)
    if not frames: return pd.DataFrame()
    df=pd.concat(frames,axis=1).sort_index()
    df=df.interpolate(limit=2).ffill().bfill()
    return df
