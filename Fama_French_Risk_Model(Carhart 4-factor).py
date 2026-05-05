import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import yfinance as yf
import requests
from io import StringIO
# market index data use
#structure like functions

# Function definitions:
# ---------------------------------------------------------------------------------------------------------------------
def marketindex_close(ticker, start_date, end_date):
    """
    Download fallback close-price data from Market Index.
    Uses fake user to avoid 403 error as marketindex blocks non-browser user agents
    """
    # This section doesnt work as marketindex still blocks the request, so need manual download
    url = f"https://www.marketindex.com.au/download-historical-data/{ticker}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.loc[start_date:end_date]
    df = df.sort_index()
    # keeps only close because only data needed
    close = df["Close"]
    return close

def get_data_returns(tickers, start_date, end_date):
    data = yf.download(
    tickers,
    start=start_date,
    end=end_date
    )
    data["Close"] = data["Close"].ffill()
    data["Volume"] = data["Volume"].fillna(0)
    # check each ticker for missing close data
    for t in tickers:
        # if missing data, use marketindex csv as fallback
        if data["Close"][t].isna().any():
            print(f"Using Market Index fallback for price data for {t}")
            # remove .AX
            mi_ticker = t.replace(".AX", "")
            try:
                fallback_close = marketindex_close(mi_ticker, start_date, end_date)
                # align dates
                fallback_close = fallback_close.reindex(
                    data.index
                )
                # fill ONLY missing values
                data["Close"][t] = data["Close"][t].fillna(
                    fallback_close
                )
            except Exception as e:
                print(f"Fallback failed for {t}: {e} - Visit https://www.marketindex.com.au/download-historical-data/{mi_ticker} to manually download csv and add to data folder")
    returns = data["Close"].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return data, returns
 

# Load and clean factor exposure data from Fama French Data Library
def load_factors(s, e, returns):
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
    factors = factors.loc[s:e]
    comm_dates = returns.index.intersection(factors.index)
    returns = returns.loc[comm_dates]
    factors = factors.loc[comm_dates]

    # Excess returns, becuase regression is a + B(MKT-RF) + S(SMB) + H(HML) + M(MOM) + e
    excess = returns.subtract(factors["RF"], axis=0)
    x = factors[["Mkt-RF", "SMB", "HML", "MOM"]]
    return x, excess


# Function to run factor/return regression to find betas(exposure) and idiosyncratic variance var(ei)
def run_factor_regression(x, excess):
    x_clean_df = x.copy()
    x_clean_df["const"] = 1

    betas = {}
    idio = {}

    for stock in excess.columns:
        df = pd.concat([excess[stock], x_clean_df], axis=1).dropna()
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        betas[stock] = beta[:-1]  # remove const
        eps = y - X @ beta
        idio[stock] = np.var(eps)

    B = pd.DataFrame(betas, index=x.columns).T

    # drop intercept if it exists
    B = B.drop(columns=["const"], errors="ignore")

    print("Beta Matrix (B):")
    print(B)
    F = x.drop(columns=["const"], errors="ignore").cov()
    idio_var = pd.Series(idio).reindex(B.index)
    D = np.diag(idio_var.values)
    B_matrix = B.values
    return B, B_matrix, F, D


# Risk decomposition 
def risk_decompos(w, B, B_matrix, F, D):
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
    print("Total Risk:", total_risk, "\n")

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

    return factor_risk, idio_risk, total_risk

# Main execution flow 
#----------------------------------------------------------------------------------------------------------------------------
# get data and returns
tickers = input("Enter the ticker symbol for portfolio separated by spaces (etc. PLY.AX LAU.AX)").split()
s = input("Enter the start date (YYYY-MM-DD): ")
e = input("Enter the end date (YYYY-MM-DD): ")
w = np.array([float(x) for x in input("Enter the portfolio weights corresponding to the order for each stock, separated by spaces (etc. 0.1336 0.1947): ").split()])

get_data_returns(tickers, s, e)
data, returns = get_data_returns(tickers, s, e)
x, excess = load_factors(s, e, returns)
B, B_matrix, F, D = run_factor_regression(x, excess)
risk_decompos(w, B, B_matrix, F, D)
