import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def run_risk_model(tickers, indicator, industry_map):
    data = yf.download(tickers, period='5y', interval='1d')['Close']
    daily_log_returns  = np.log(data / data.shift(1))
    monthly_prices     = data.resample('ME').last()
    monthly_returns    = np.log(monthly_prices / monthly_prices.shift(1))[1:]
    monthly_volatility = daily_log_returns.resample('ME').std()[1:]


    if indicator == 1:
        momentum = monthly_returns.rolling(window=3).sum().shift(1)
        volume_data = yf.download(tickers, period='5y', interval='1d')['Volume']
        liquidity   = np.log(volume_data.resample('ME').sum())

        df = pd.concat({
            'return':             monthly_returns,
            'vol':                monthly_volatility,
            'momentum':           momentum,
            'liquidity':          liquidity,
        }, axis=1).dropna()

        industries = sorted(set(industry_map.values()))
        industry_factors = {}
        for ind in industries:
            dummy = pd.DataFrame(0, index=monthly_returns.index, columns=monthly_returns.columns)
            for stock in dummy.columns:
                if industry_map.get(stock) == ind:
                    dummy.loc[:, stock] = 1
            industry_factors[f'ind_{ind}'] = dummy


        cs_mean  = monthly_prices.mean(axis=1)
        cs_std   = monthly_prices.std(axis=1, ddof=1).replace(0, np.nan)
        price_std = monthly_prices.sub(cs_mean, axis=0).div(cs_std, axis=0)

        raw_factors = {
            'vol':       df['vol'],
            'momentum':  df['momentum'],
            'liquidity': df['liquidity'],
            'size':      price_std,
            'return':    df['return'],
        }
        df_std = {}
        for factor, raw in raw_factors.items():
            if factor == 'return' or factor == 'size':
                df_std[factor] = raw
            else:
                m = raw.mean(axis=1)
                s = raw.std(axis=1, ddof=1)
                df_std[factor] = raw.sub(m, axis=0).div(s, axis=0)

        target_index = df_std['return'].index
        df_std['size'] = df_std['size'].loc[target_index]
        for ind in industries:
            industry_factors[f'ind_{ind}'] = industry_factors[f'ind_{ind}'].loc[target_index]

        betas_z      = pd.DataFrame()
        r_sq_z       = {}
        cs_residuals = {}

        for stock in tickers:
            y = df_std['return'][stock]
            X = pd.concat(
                [df_std['vol'][stock], df_std['momentum'][stock],
                 df_std['liquidity'][stock], df_std['size'][stock]]
                + [industry_factors[f'ind_{ind}'][stock] for ind in industries[:-1]],
                axis=1
            )
            X.columns = ['vol', 'momentum', 'liquidity', 'size'] + industries[:-1]
            X = sm.add_constant(X)
            model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            betas_z[stock]      = model.params
            r_sq_z[stock]       = model.rsquared
            cs_residuals[stock] = model.resid

        betas_z      = betas_z.T
        r_sq_z       = pd.Series(r_sq_z)
        cs_residuals = pd.DataFrame(cs_residuals, index=target_index)

        print("=== Betas ===")
        print(betas_z)
        print("\n=== R² ===")
        print(r_sq_z)

        factor_returns_z = {}
        for month in target_index:
            y = df_std['return'].loc[month]
            X = pd.concat(
                [df_std['vol'].loc[month], df_std['momentum'].loc[month],
                 df_std['liquidity'].loc[month], df_std['size'].loc[month]]
                + [industry_factors[f'ind_{ind}'].loc[month] for ind in industries[:-1]],
                axis=1
            )
            X.columns = ['vol', 'momentum', 'liquidity', 'size'] + industries[:-1]
            X, y = X.align(y, join='inner', axis=0)
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            factor_returns_z[month] = model.params

        factor_returns_z = pd.DataFrame(factor_returns_z).T
        print("\n=== Factor Returns ===")
        print(factor_returns_z)

        cov_matrix_z = factor_returns_z.drop(columns=['const']).cov()
        print("\n=== Factor Covariance Matrix ===")
        print(cov_matrix_z)

        idio_var = cs_residuals.var(ddof=1)
        D = np.diag(idio_var.values)
        print("\n=== Idiosyncratic Variance ===")
        print(idio_var)

        B = betas_z[cov_matrix_z.columns].values
        Sigma = B @ cov_matrix_z.values @ B.T + D
        Sigma_df = pd.DataFrame(Sigma, index=tickers, columns=tickers)
        print("\n=== Stock Covariance Matrix ===")
        print(Sigma_df)

        window       = 12
        factors_only = factor_returns_z.drop(columns=['const'] + [
            c for c in factor_returns_z.columns
            if c in industries or c == 'const'
        ])
        rolling_cov  = factors_only.rolling(window).cov()
        factor_names = factors_only.columns
        dates        = factors_only.index[window - 1:]

        cov_series = {}
        for i in range(len(factor_names)):
            for j in range(i, len(factor_names)):
                f1, f2 = factor_names[i], factor_names[j]
                cov_series[f'{f1}-{f2}'] = [rolling_cov.loc[d].loc[f1, f2] for d in dates]

        plt.figure(figsize=(12, 7))
        for name, series in cov_series.items():
            plt.plot(dates, series, label=name)
        plt.title('Factor Covariances Over Time')
        plt.ylabel('Covariance')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    elif indicator == 0:

        def load_ff_daily(path, skiprows, nrows=None, col_names=None):
            df = pd.read_csv(
                path, skiprows=skiprows, nrows=nrows,
                index_col=0, na_values=['-99.99', '-999']
            )
            df.index = pd.to_datetime(
                df.index.astype(str).str.strip(), format='%Y%m%d', errors='coerce'
            )
            df = df[df.index.notna()]
            df.index.name = 'Date'
            if col_names:
                df.columns = col_names
            else:
                df.columns = df.columns.str.strip()
            df = df.apply(pd.to_numeric, errors='coerce') / 100
            df = df.dropna(how='all').resample('ME').sum()
            return df

        ap_factors  = load_ff_daily('ap_factors.csv', skiprows=6)
        industry_vw = load_ff_daily(
            'ind.csv', skiprows=10, nrows=26192,
            col_names=['Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other'])

        common_index = monthly_returns.index \
            .intersection(ap_factors.index) \
            .intersection(industry_vw.index)

        stock_ret    = monthly_returns.loc[common_index]
        ff_factors   = ap_factors.loc[common_index, ['Mkt-RF', 'SMB', 'HML']]
        rf           = ap_factors.loc[common_index, 'RF']
        ind_ret_all  = industry_vw.loc[common_index]
        stock_excess = stock_ret.sub(rf, axis=0)

        print(f"Aligned dataset: {len(common_index)} monthly observations")
        print(f"Date range: {common_index[0].date()} → {common_index[-1].date()}")

        betas_n     = {}
        r_squared_n = {}
        residuals_n = {}

        for stock in tickers:
            ind_col = industry_map[stock]
            y = stock_excess[stock].dropna()
            X = pd.concat([
                ff_factors[['Mkt-RF', 'SMB', 'HML']],
                ind_ret_all[ind_col].rename('Ind')
            ], axis=1).loc[y.index]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            betas_n[stock]     = model.params
            r_squared_n[stock] = model.rsquared
            residuals_n[stock] = model.resid

        betas_df_n = pd.DataFrame(betas_n).T
        r_sq_n     = pd.Series(r_squared_n, name='R²')
        print("\n=== Betas ===")
        print(betas_df_n.round(4))
        print("\n=== R² ===")
        print(r_sq_n.round(4))

        used_industries = list(set(industry_map.values()))
        factor_ret = pd.concat([ff_factors, ind_ret_all[used_industries]], axis=1)
        F = factor_ret.cov()
        print("\n=== Factor Covariance Matrix ===")
        print(F.round(8))

        resid_df  = pd.DataFrame(residuals_n)
        idio_var_ = resid_df.var(ddof=1)
        D = np.diag(idio_var_.values)
        print("\n=== Idiosyncratic Variance ===")
        print(idio_var_.round(8))

        factor_cols = list(F.columns)
        B_rows = []
        for stock in tickers:
            ind_col = industry_map[stock]
            row = []
            for fc in factor_cols:
                if fc == ind_col:
                    row.append(betas_df_n.loc[stock, 'Ind'])
                elif fc in betas_df_n.columns:
                    row.append(betas_df_n.loc[stock, fc])
                else:
                    row.append(0.0)
            B_rows.append(row)

        B = np.array(B_rows)
        Sigma = B @ F.values @ B.T + D
        Sigma_df = pd.DataFrame(Sigma, index=tickers, columns=tickers)
        print("\n=== Full Covariance Matrix Σ (annualised ×12) ===")
        print((Sigma_df * 12).round(6))

        window = 12
        rolling_cov  = factor_ret[['Mkt-RF', 'SMB', 'HML']].rolling(window).cov()
        factor_names = ['Mkt-RF', 'SMB', 'HML']
        dates = factor_ret.index[window - 1:]

        cov_series = {}
        for i in range(len(factor_names)):
            for j in range(i, len(factor_names)):
                f1, f2 = factor_names[i], factor_names[j]
                cov_series[f'{f1}–{f2}'] = [rolling_cov.loc[d].loc[f1, f2] for d in dates]

        plt.figure(figsize=(13, 5))
        for name, series in cov_series.items():
            plt.plot(dates, series, label=name)
        plt.ylabel('Covariance')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("indicator must be 0 or 1")