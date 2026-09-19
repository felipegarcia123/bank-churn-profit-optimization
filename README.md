# Retention Copilot

**Profit-driven bank churn prediction and retention campaign prioritization.**

Retention Copilot is an end-to-end machine learning application that transforms customer churn predictions into **actionable retention decisions**.

Instead of selecting a model solely based on predictive performance, the system combines calibrated churn probabilities with estimated customer value, campaign cost, and assumed retention success to prioritize customers by **expected incremental profit**.

### Key Results

* **10,000 customers** analyzed
* **2,000 customers** in the held-out test set
* **XGBoost** selected through validation-based economic evaluation
* **ROC-AUC:** 0.855
* **Average Precision:** 0.698
* **Brier Score:** 0.104
* **31.5% fewer contacts** than contacting every customer
* **US$6,405 additional simulated incremental profit** compared with contacting everyone
* **1,353% simulated incremental ROI**

> **Important:** Financial results are simulations based on explicit assumptions. They do not represent observed revenue or proven causal effects.

`Python` · `Scikit-learn` · `XGBoost` · `Streamlit` · `Machine Learning` · `Financial Analytics`

---

## Business Problem

Customer churn models are often evaluated using metrics such as ROC-AUC, accuracy, or F1-score.

However, the model with the strongest predictive metric is not necessarily the model that produces the best **business decision**.

A retention team faces a practical question:

> **Which customers should we contact to maximize the expected financial benefit of a retention campaign?**

Retention Copilot addresses this problem by combining:

* churn probability;
* estimated annual customer value;
* retention campaign cost;
* assumed retention success rate.

The result is a ranked list of customers based on their **expected incremental profit from being contacted**.

The Streamlit application recalculates this prioritization dynamically when campaign cost or assumed success rate changes.

---

## Business Decision Framework

The system estimates **annual customer revenue**, not full customer lifetime value (CLV):

```text
customer_value =
    0.025 × balance
    + 80 × number_of_products
    + 150 × credit_card_indicator
```

These coefficients are illustrative business assumptions.

Customer balance is capped using the 99th percentile learned from the training data to reduce the influence of extreme values.

For each custom
