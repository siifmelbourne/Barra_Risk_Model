import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import yfinance as yf

# get data and returns
# enter tickers if generalising function
tickers = [
    "PLY.AX", "LAU.AX", "TLX.AX", "COS.AX", 
    "ANG.AX", "VVA.AX", "WTC.AX", "AUB.AX", "XYZ.AX", "DUG.AX"
    ]
data = yf.download(tickers, start ="2024-01-01", end = "2025-12-31")
#missing XYZ.AX data
data["Close"] = data["Close"].ffill()
data["Volume"] = data["Volume"].fillna(0)
returns = data["Close"].pct_change()
returns = returns.replace([np.inf, -np.inf], np.nan)
returns = returns.loc["2025-01-23":"2025-12-31"] #Change this if exclude XYZ

# Load and clean Fama-french data
f3_data = pd.read_csv(
    'C:/Users/tiany/OneDrive/Attachments/Barra_Risk_Model/Asia_Pacific_ex_Japan_3_Factors_Daily.csv',
    skiprows=6,
    skipfooter=1,
    engine="python"
)

mom = pd.read_csv(
    'C:/Users/tiany/OneDrive/Attachments/Barra_Risk_Model/Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv',
    skiprows=6,
    skipfooter=1,
    engine="python")
f3_data.columns = f3_data.columns.str.strip()
date_col = f3_data.columns[0]
f3_data = f3_data[f3_data[date_col].astype(str).str.match(r"^\d{8}$")]
f3_data[date_col] = pd.to_datetime(f3_data[date_col], format="%Y%m%d")
f3_data = f3_data.set_index(date_col)
f3_data = f3_data.apply(pd.to_numeric, errors="coerce") / 100

mom.columns = mom.columns.str.strip()
date_col = mom.columns[0]
mom = mom[mom[date_col].astype(str).str.match(r"^\d{8}$")]
mom[date_col] = pd.to_datetime(mom[date_col], format="%Y%m%d")
mom = mom.set_index(date_col)
mom = mom.apply(pd.to_numeric, errors="coerce") / 100

# join factors (used Fama French 3 factors + momentum factor (carhart?) separate csv, as this aligns best with the factors we did before)
factors = f3_data.join(mom, how='inner')
# Winners - Losers = Momentum factor so rename
factors = factors.rename(columns = {"WML":"MOM"})

# industry factor?

# Now select date range
# (XYZ is still missing price data! to mitigat this, either:
# 1. Remove XYZ from analysis and use updated weights []
# 2. Use last available date range 
factors = factors.loc["2025-01-23":"2025-12-31"]
comm_dates = returns.index.intersection(factors.index)
returns = returns.loc[comm_dates]
factors = factors.loc[comm_dates]

# Excess returns, becuase regression is a + B(MKT-RF) + S(SMB) + H(HML) + M(MOM) + e
excess = returns.subtract(factors["RF"], axis=0)
x = factors[["Mkt-RF", "SMB", "HML", "MOM"]]

# Run factor/return regression to find betas(exposure) and idiosyncratic variance var(ei)

x_reg = x.copy()
x_reg["const"] = 1   # intercept for regression
factor_names = x.columns.tolist()

betas = {}
idio = {}
for stock in excess.columns:
    y = excess[stock]
    df = pd.concat([y, x_reg], axis=1).dropna()
    y_clean = df.iloc[:, 0].values
    x_clean = df.iloc[:, 1:].values
    beta = np.linalg.lstsq(x_clean, y_clean, rcond=None)[0]
    y_hat = x_clean @ beta
    eps = y_clean - y_hat
    betas[stock] = beta[:-1]   # remove intercept cleanly
    idio[stock] = np.var(eps)


B = pd.DataFrame(betas, index=x_reg.columns).T
# drop intercept if it exists
B = B.drop(columns=["const"], errors="ignore")

print("Beta Matrix (B):")
print(B)

F = x.drop(columns=["const"], errors="ignore").cov()
idio_var = pd.Series(idio).reindex(B.index)
D = np.diag(idio_var.values)

w = np.array([
    0.1336, 0.0947, 0.0414, 0.0788, 0.0659,
    0.0516, 0.0491, 0.0464, 0.0261, 0.0252
]) #change thise if excluding XYZ to [0.1485, 0.1053, 0.0461, 0.0878, 0.0732, 0.0573, 0.0544, 0.0515, 0.0289]
B_matrix = B.values

# Risk decomposition
factor_risk = w.T @ B_matrix @ F.values @ B_matrix.T @ w
idio_risk = w.T @ D @ w
total_risk = factor_risk + idio_risk


print("\nFactor Risk:", factor_risk)
print("Idiosyncratic Risk:", idio_risk)
print("Total Risk:", total_risk)
F_matrix = F.values
factor_risk = w.T @ B_matrix @ F_matrix @ B_matrix.T @ w
idio_risk = w.T @ D @ w
total_risk = factor_risk + idio_risk
print("Factor Risk:", factor_risk)
print("Idiosyncratic Risk:", idio_risk)
print("Total Risk:", total_risk)

#decomposing volatility:
factor_var = w.T @ B @ F @ B.T @ w
idio_var_port = w.T @ D @ w
total_var = factor_var + idio_var_port
factor_vol = np.sqrt(factor_var)
idio_vol = np.sqrt(idio_var_port)
total_vol = np.sqrt(total_var)
print("Factor Volatility:", factor_vol)
print("Idiosyncratic Volatility:", idio_vol)
print("Total Portfolio Volatility:", total_vol)
