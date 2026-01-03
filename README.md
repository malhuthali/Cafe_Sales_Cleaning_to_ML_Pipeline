# ☕ Cafe Sales: Data Recovery & ML Pipeline

## 📊 Project Overview

This project focuses on **recovering data correctness from a deliberately dirty cafe transaction dataset**. The core objective is to **restore data integrity before modeling** by applying a **two-stage recovery strategy**:

1.  **Deterministic recovery using math** (Ground Truth).
2.  **Statistical recovery** (Imputation) only when deterministic reconstruction is impossible.

Beyond recovery, the pipeline applies **outlier control**, **target normalization**, and **encoding** to prepare a stable modeling dataset. Two tree-based models (Random Forest & XGBoost) are then used to validate whether the recovered data behaves consistently under feature ablation.

This project is therefore primarily a **data cleaning, validation, and recovery exercise**, with modeling used as a diagnostic tool rather than the end goal.

**Dataset Source:**
[Cafe Sales Dirty Data (Kaggle)](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training)

---

## 📁 1. Data Auditing & Cleaning

**Goal:** Identify corruption, standardize formats, and restore mathematical ground truth.

* **Initial Audit & Standardization:**
    * Converted invalid placeholders (`'ERROR'`, `'UNKNOWN'`) to `NaN`.
    * Enforced strict data types (numeric/datetime) to expose true missingness.
    * **Missing Data Analysis:** Generated summary tables and graphs to visualize the extent of data loss.

* **Categorical Cleaning:**
    * **Items:** Null items (9.69%) were filled with `"Other"`.
    * **High Missingness Columns:** `Location` (39.6% missing) and `Payment Method` (31.8% missing) were filled with `"Not Recorded"` to preserve row integrity rather than dropping valuable data.
    * **Transaction Date:** Rows with missing dates (4.6%) were dropped as they constituted a negligible portion of the data.

* **The Two-Stage Recovery Strategy:**
    * **Stage 1: Deterministic Recovery (Math):**
        I utilized the formula `Total Spent = Quantity * Price` to recover missing values with 100% certainty.

    ```python
    # 1. Recover Total Spent (T = Q * P)
    mask_spent = df['Total Spent'].isna() & df['Quantity'].notna() & df['Price Per Unit'].notna()
    df.loc[mask_spent, 'Total Spent'] = df['Quantity'] * df['Price Per Unit']

    # 2. Recover Quantity (Q = T / P)
    mask_qty = df['Quantity'].isna() & df['Total Spent'].notna() & df['Price Per Unit'].notna()
    df.loc[mask_qty, 'Quantity'] = df['Total Spent'] / df['Price Per Unit']

    # 3. Recover Price (P = T / Q)
    mask_price = df['Price Per Unit'].isna() & df['Total Spent'].notna() & df['Quantity'].notna()
    df.loc[mask_price, 'Price Per Unit'] = df['Total Spent'] / df['Quantity']
    ``` 
    * **Stage 2: Statistical Recovery (Imputation):**
            For rows where deterministic recovery was impossible (missing ≥ 2 variables, roughly 58 rows), I imputed values using **Grouped Medians** (median Price/Quantity per Item).
        

* **Validation Checks:**
    * **Math Consistency:** Confirmed 0 violations of `Total = Quantity × Price`.
    * **Distribution Stability:** Average spend remained stable (Imputed: 8.87 vs Non-Imputed: 8.92).


---

## 📈 2. Exploratory Data Analysis (EDA)

**Goal:** Understand the "shape" of the business and handle anomalies.

* **Distribution Analysis:** Checked for **Skewness** to understand data shape; `Total Spent` showed high right-skewness, indicating the presence of extreme transaction values.
    
* **Outlier Detection:** Used **IQR** and **Z-Score** methods to mathematically identify observations falling outside normal bounds.
    
* **Outlier Management:** Chose **IQR filtering** to remove extreme noise, as it is more robust for skewed distributions than Z-Score.

---

## 🛠️ 3. Feature Engineering

**Goal:** Prepare the data for machine learning.

* **Time-Based Features:** Extracted granular time signals: `Year`, `Month`, `Day`, `Weekday`, `Hour`, `Is_Weekend`, and `Season`.
* **Target Transformation:** `Total Spent` showed high skewness (> 0.5). I applied a Log Transformation (`Total Spent Log`) to normalize the target for regression.
* **Encoding:** Applied One-Hot Encoding to categorical columns: `['Item', 'Payment Method', 'Location', 'season']`.
* **Scaling:** Applied `StandardScaler` to `Quantity` and `Price Per Unit` (fit on train, transform on test).

---

## 🧪 4. Modeling & Feature Ablation

**Goal:** Use Random Forest and XGBoost to diagnose the predictive "signal" in the data.

The models were tested on `Total Spent Log` under four scenarios to measure feature importance.

### 1️⃣ All Features (Baseline)

_Validates that the pipeline preserved the mathematical relationship._

|**Model**|**R²**|**RMSE**|
|---|---|---|
|**Random Forest**|**0.9975**|**0.0308**|
|**XGBoost**|**0.9973**|**0.0324**|

### 2️⃣ No Price

_Can the model guess revenue based on Item + Quantity?_

|**Model**|**R²**|**RMSE**|
|---|---|---|
|**Random Forest**|0.9497|0.1393|
|**XGBoost**|0.9356|0.1575|

### 3️⃣ No Quantity

_Can the model guess revenue based on Item + Price?_

|**Model**|**R²**|**RMSE**|
|---|---|---|
|**Random Forest**|0.3267|0.5095|
|**XGBoost**|0.3113|0.5153|

### 4️⃣ No Quantity & No Price (Context Only)

_Can the model guess revenue based on Time + Location + Item?_

|**Model**|**R²**|**RMSE**|
|---|---|---|
|**Random Forest**|0.2690|0.5309|
|**XGBoost**|0.2667|0.5317|

---

### 📊 Feature Importance Analysis

The ablation results above are explained by the feature importance scores. `Quantity` and `Price Per Unit` account for **over 99%** of the predictive signal.

| **Feature**                  | **Importance Score** | **Contribution**      |
| ---------------------------- | -------------------- | --------------------- |
| **Quantity**                 | **0.6355**           | **63.5% (Dominant)**  |
| **Price Per Unit**           | **0.3604**           | **36.0% (Secondary)** |
| `is_imputed`                 | 0.0027               | < 0.3%                |
| `season_Summer`              | 0.0005               | Negligible            |
| `Payment Method_Credit Card` | 0.0002               | Negligible            |

---

## 🧠 Final Conclusions

* **Data Integrity Success:** The near-perfect baseline R² (0.9975) confirms the cleaning and recovery strategy was successful and mathematically sound.
* **The "Quantity" Driver:** The ablation study proves that **Quantity is the single most critical variable**. Removing it causes performance to collapse (R² drops from ~0.99 to ~0.32).
* **Price Inference:** The model can reasonably infer `Price` from the `Item` category (maintaining ~0.95 R²), but it cannot infer `Quantity` from context alone.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt

```
