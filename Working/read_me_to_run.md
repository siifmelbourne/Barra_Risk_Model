# Factor Risk Model — How to Run

This guide explains how to set up and run the Python script for portfolio risk decomposition using a factor model.

---

## 1. Requirements

Make sure you have **Python 3.9+** installed.

### Install required packages

Run this in your terminal:

```bash
pip install numpy pandas matplotlib yfinance requests
```

---

## 2. Save the Script

Save your code into a Python file, for example:

```bash
Fama_French_Barra_Risk_Model.py
```

---

## 3. Running the Script

In your terminal (or VS Code terminal), run:

```bash
python Fama_French_Barra_Risk_Model.py
```

---

## 4. Required Inputs

When prompted, enter the following:

### 1. Tickers

Enter stock tickers separated by spaces:

```
PLY.AX LAU.AX WTC.AX
```

---

### 2. Start Date

Format: `YYYY-MM-DD`

```
2021-01-01
```

---

### 3. End Date

```
2024-01-01
```

---

### 4. Portfolio Weights

Weights must:

* Match ticker order
* Sum to **1**

Example:

```
0.3 0.4 0.3
```

---
## 5. Notes & Troubleshooting

### Missing Data

If you see:

```
Using Market Index fallback for price data
```

* The fallback may fail due to website blocking
* Manually download CSV from:

```
https://www.marketindex.com.au/download-historical-data/___insert_stock_name__
```

---
