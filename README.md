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

For each customer:

```text
expected_benefit_of_contact =
    P(churn) × customer_value × retention_success_rate − contact_cost
```

Only contacts with a positive expected benefit are recommended, ranked by that benefit. With constant cost and success rate this is equivalent to ranking by probability × value. The optional minimum-risk filter in the app is an additional filter, not the main criterion.

Retrospective incremental profit is calculated as `success_rate × value of contacted churners − total cost`. Not contacting has zero incremental profit. Revenue that remains at risk is reported separately and is not subtracted again. ROI divides incremental profit by cost; **maximizing profit is not the same as maximizing ROI**.

---

## Methodology

1. Reproducible stratified split: **60% train / 20% validation / 20% test**.
2. Three candidates: Logistic Regression, Random Forest and XGBoost.
3. Three-fold sigmoid calibration within the training set. Each fold also fits imputation, p99 caps, scaling and encoding.
4. Model selection by the economic policy's profit on validation. No global threshold is tuned on test.
5. Evaluation of the selected model on the held-out test set, compared with no action, contacting everyone and a random selection with the same coverage.

The random baseline uses its analytical expectation when selecting the same number of customers without replacement, avoiding variation from a single seed. Transformations are persisted: a customer receives the same processing regardless of the scoring batch size.

---

## Results of the Included Run

Dataset: 10,000 rows; test set: **2,000 customers**, 407 churners. Contact cost: **US$15**; assumed retention success: **30%**. All rows use the same test set.

| Policy | Contacts | Cost | Simulated incremental profit |
|---|---:|---:|---:|
| No action | 0 | US$0 | US$0 |
| Contact everyone | 2,000 | US$30,000 | US$271,627.87 |
| Random, same coverage (expected) | 1,370 | US$20,550 | US$186,065.09 |
| XGBoost + economic policy | 1,370 | US$20,550 | **US$278,032.87** |

Improvement over contacting everyone: **US$6,405** with **31.5% fewer contacts**. ROC-AUC: **0.855**; Average Precision: **0.698**; Brier score: **0.104**. Simulated incremental ROI: **1,353%**.

XGBoost beat Random Forest by only US$54 on validation. That small difference does not demonstrate robust superiority; these figures come from a single fixed split and do not include confidence intervals.

Values and the dataset hash are in [the JSON report](reports/executive_report.json). They should not be compared directly with earlier versions that selected the model on test. The app scores the full local dataset, which may include training rows, so its figures do not replace this held-out evaluation.

---

## Getting Started

From the project root, with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place `Churn_Modelling.csv` in `data/raw/`. The source is [Bank Customer Churn on Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling). The data is not distributed with this repository.

```bash
python -m src.train_pipeline
streamlit run app.py
```

The app reads the CSV directly from the folder; no browser upload is required. It accepts the original Kaggle schema or the normalized names defined in `src/data_processing.py`. For scoring, `Exited` is optional; the ten original predictor columns are required. `CustomerId` is optional and, if present, must be unique and non-empty.

Direct dependency versions are pinned for Python 3.11. `xgboost` is pinned to 3.2.0, the latest PyPI release that supports Python 3.11; the included report was generated with the versions recorded in `reports/executive_report.json`, so retraining with the pinned versions may produce small numeric differences. No lockfile of transitive dependencies is included. On some systems XGBoost requires a compatible OpenMP runtime.

Alternative with Conda, including notebook tools:

```bash
conda env create -f environment.yml
conda activate churn
```

---

## Tests and Notebook

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
jupyter notebook notebooks/01_churn_financial_analysis.ipynb
```

The tests use synthetic data and temporary artifacts: they do not require downloading the dataset or training the real model. CI runs the same tests on Python 3.11. The notebook reads the aggregated report; it does not reselect models on test.

---

## Project Structure

```text
app.py                         Dashboard and local data loading
src/data_processing.py         Validation, split and persisted transformations
src/modeling.py                Classifiers and calibration
src/financial_evaluator.py     Policy, profit, baselines and ranking
src/train_pipeline.py          Reproducible training and export
tests/                         Unit and Streamlit tests
notebooks/                     Explanatory reading of results
reports/executive_report.*     Versioned aggregated results
data/raw/                      Local dataset, excluded from Git
models/                        Local artifact, excluded from Git
docs/PORTFOLIO.md              Case study
docs/PUBLISHING.md             Git and publishing notes
.github/workflows/ci.yml       Tests on every push and pull request
```

`reports/priority_calls.csv` contains only test-set recommendations and is excluded from Git. The CSV downloaded from the app corresponds to the local dataset being scored.

---

## Configuration and Maintenance

```bash
python -m src.train_pipeline --retention-cost 25 --retention-success 0.40
```

The CLI accepts paths through `--data`, `--out-model`, `--out-report` and `--out-calls`. The app accepts `CHURN_DATA_PATH` and `CHURN_MODEL_PATH` as environment variables; by default it resolves paths relative to `app.py`. `.env.example` documents these options; `.env` is not loaded automatically.

After changing Python modules, restart the server with Ctrl+C and `streamlit run app.py`; reloading only the browser can keep previous modules in memory. When the model is regenerated, the app invalidates its cache based on the file modification time. v1 artifacts require retraining.

---

## Limitations

- There is no intervention data: retention success is a homogeneous assumption, not a causal estimate.
- Annual value omits lifetime horizon, discounting, full costs and offer heterogeneity.
- Validation uses a random split; it does not demonstrate temporal stability or generalization to another bank.
- Calibration is fitted with cross-validation but needs further evaluation by segment and out of sample.
- The model includes geography and gender. A fairness assessment for operational use has not been completed.
- The app is local, without authentication or controls for a real production customer base.

---

## Author

**Juan Felipe García**
https://www.linkedin.com/in/juan-felipe-garc%C3%ADa-garc%C3%ADa-9a167912a/
https://github.com/felipegarcia123
