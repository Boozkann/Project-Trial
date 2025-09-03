# features.py
import pandas as pd, numpy as np
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    if {'exports_gdp','imports_gdp'}.issubset(out.columns):
        out['trade_openness']=out['exports_gdp']+out['imports_gdp']
    if 'gdp_growth' in out.columns:
        out['gdp_growth_3y_avg']=out['gdp_growth'].rolling(3,min_periods=1).mean()
    if 'inflation_cpi' in out.columns:
        out['inflation_vol_5y']=out['inflation_cpi'].rolling(5,min_periods=2).std()
    if {'central_govt_debt_gdp','lending_rate'}.issubset(out.columns):
        out['debt_x_lendrate']=out['central_govt_debt_gdp']*out['lending_rate']
    for col in ['central_govt_debt_gdp','inflation_cpi','cash_surplus_deficit_gdp','gdp_growth','reserves_months_imports']:
        if col in out.columns: out[f'{col}_lag1']=out[col].shift(1)
    if 'central_govt_debt_gdp' in out.columns:
        out['debt_gdp_gt_90']=(out['central_govt_debt_gdp']>90).astype(int)
    if 'reserves_months_imports' in out.columns:
        out['reserves_lt_3m']=(out['reserves_months_imports']<3).astype(int)
    return out.dropna(how='all')
