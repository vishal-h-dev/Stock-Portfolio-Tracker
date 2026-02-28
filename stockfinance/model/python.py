import yfinance as yf
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model and label encoder
model = load("xgboost_stock_model.joblib")       # rename your first file to model.joblib
label_encoder = load("label_encoder.joblib")  # rename your second file to encoder.joblib

# Helper functions for feature extraction (same as training)
def clean_value(val, min_val=None, max_val=None):
    if val is None or pd.isna(val):
        return np.nan
    try:
        val = float(val)
        if min_val is not None and val < min_val:
            return np.nan
        if max_val is not None and val > max_val:
            return np.nan
        return val
    except:
        return np.nan

def get_value_or_nan(df, possible_rows):
    for row in possible_rows:
        if row in df.index:
            try:
                val = df.loc[row]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.dropna()
                    if len(val) == 0:
                        continue
                    val = val.iloc[-1]
                if pd.notna(val):
                    return val
            except:
                continue
    return np.nan

def get_rd_expense(financials, cashflow):
    keys = ["Research Development", "ResearchAndDevelopment", "R&D Expense"]
    val = get_value_or_nan(cashflow, keys)
    if pd.isna(val):
        val = get_value_or_nan(financials, keys)
    return val

def get_sector_industry(info):
    return info.get('sector', "Unknown"), info.get('industry', "Unknown")

def get_stock_features(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials if not stock.financials.empty else stock.quarterly_financials
    balance_sheet = stock.balance_sheet if not stock.balance_sheet.empty else stock.quarterly_balance_sheet
    cashflow = stock.cashflow if not stock.cashflow.empty else stock.quarterly_cashflow

    net_income = get_value_or_nan(financials, ["Net Income"])
    equity = get_value_or_nan(balance_sheet, ["Total Stockholder Equity"])
    current_assets = get_value_or_nan(balance_sheet, ["Total Current Assets"])
    current_liabilities = get_value_or_nan(balance_sheet, ["Total Current Liabilities"])
    operating_cf = get_value_or_nan(cashflow, ["Total Cash From Operating Activities"])
    total_assets = get_value_or_nan(balance_sheet, ["Total Assets"])
    total_debt = get_value_or_nan(balance_sheet, ["Total Debt"])
    gross_profit = get_value_or_nan(financials, ["Gross Profit"])
    total_revenue = get_value_or_nan(financials, ["Total Revenue"])
    ebitda = get_value_or_nan(financials, ["EBITDA"])
    rd_expense = clean_value(get_rd_expense(financials, cashflow), min_val=0)

    pe = clean_value(info.get("trailingPE", np.nan), 0, 100)
    pb = clean_value(info.get("priceToBook", np.nan), 0, 50)
    debt_to_equity = clean_value(info.get("debtToEquity", np.nan), 0, 500)
    dividend_yield = clean_value(info.get("dividendYield", np.nan), 0, 1)
    revenue_growth = clean_value(info.get("revenueGrowth", np.nan), -1, 2)
    eps_growth = clean_value(info.get("earningsGrowth", np.nan), -1, 2)
    market_cap = clean_value(info.get("marketCap", np.nan), 0)

    gross_margin = gross_profit / total_revenue if total_revenue else np.nan
    ebitda_margin = ebitda / total_revenue if total_revenue else np.nan
    debt_to_assets = total_debt / total_assets if total_assets else np.nan
    current_ratio = current_assets / current_liabilities if current_liabilities else np.nan
    roe = net_income / equity if equity else np.nan

    sector, industry = get_sector_industry(info)
    hist = stock.history(period="90d")
    volatility = hist['Close'].pct_change().std() * (252 ** 0.5) if not hist.empty else np.nan
    momentum_1m = hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1 if len(hist) > 21 else np.nan
    momentum_3m = hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1 if not hist.empty else np.nan

    data = {
        "Sector": sector,
        "Industry": industry,
        "P/E Ratio": pe,
        "P/B Ratio": pb,
        "Total Debt/Equity": debt_to_equity,
        "Debt/Assets": debt_to_assets,
        "Dividend Yield (%)": dividend_yield * 100 if dividend_yield else np.nan,
        "Current Ratio": current_ratio,
        "Operating Cash Flow": operating_cf,
        "Gross Margin": gross_margin,
        "EBITDA Margin": ebitda_margin,
        "Revenue Growth (YoY)": revenue_growth,
        "EPS Growth": eps_growth,
        "ROE": roe,
        "R&D Expense": rd_expense,
        "Market Cap": market_cap,
        "Volatility": volatility,
        "Momentum 1M": momentum_1m,
        "Momentum 3M": momentum_3m,
    }
    return pd.DataFrame([data])

# Get and process AAPL data
df = get_stock_features("AAPL")

# Fill missing numeric values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode sector and industry
df['Sector'] = df['Sector'].fillna("Unknown")
df['Industry'] = df['Industry'].fillna("Unknown")
df['Sector'] = encoder.transform(df['Sector'])
df['Industry'] = encoder.transform(df['Industry'])

# Predict
prediction = model.predict(df)
predicted_label = encoder.inverse_transform(prediction)
print(f"AAPL prediction: {predicted_label[0]}")
