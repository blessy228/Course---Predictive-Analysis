"""
MDI3003 Lab 04 - Customer Segment Prediction
Dataset: UCI Online Retail II (Dataset C, Research Extension per lab manual)

IMPORTANT (per manual section 7.7 / Dataset-use rule):
Online Retail II does NOT ship with predefined customer segment labels.
The manual requires a "two-stage extension with a frozen, documented label-construction
procedure" before any supervised model can be trained. That is implemented below:

STAGE 1 (label construction, frozen BEFORE modeling):
    Standard RFM (Recency, Frequency, Monetary) scoring is used to assign every customer
    to one of four business segments: Champions, Loyal Customers, At Risk, Hibernating/Lost.

STAGE 2 (supervised prediction):
    A SEPARATE, non-circular feature set (country, average unit price, bulk-buy ratio,
    average basket size, product diversity, return rate, tenure, weekend ratio) is used
    to predict the Stage 1 label. The raw R, F, M values themselves are NOT used as
    predictors, because doing so would let the model trivially reconstruct the label it
    was built from (label circularity, explicitly flagged as a common mistake in the manual).
"""

from pathlib import Path
import json, platform, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)

SEED = 42
np.random.seed(SEED)

OUT = Path("lab04_outputs")
for d in ["figures", "models", "artifacts", "results"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

versions = {
    "python": sys.version,
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
}
(OUT / "artifacts" / "versions.json").write_text(json.dumps(versions, indent=2))
print(versions)

# ---------------------------------------------------------------------------
# A2. Load and validate the dataset
# ---------------------------------------------------------------------------
DATA_PATH = Path("C:/FallSemester/C2 - Advanced Predictive Analytics/lab-DA/4online_retail_II.xlsx")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df1 = pd.read_excel(DATA_PATH, sheet_name="Year 2009-2010")
df2 = pd.read_excel(DATA_PATH, sheet_name="Year 2010-2011")
raw = pd.concat([df1, df2], ignore_index=True)
print("Raw transaction shape:", raw.shape)

dataset_card = {
    "dataset_name": "UCI Online Retail II",
    "source": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    "doi": "10.24432/C5CG6D",
    "licence": "UCI Machine Learning Repository, public research/educational use",
    "record_count_raw": int(raw.shape[0]),
    "feature_count_raw": int(raw.shape[1]),
    "date_range": [str(raw["InvoiceDate"].min()), str(raw["InvoiceDate"].max())],
    "missing_customer_id_percent": round(float(raw["Customer ID"].isna().mean() * 100), 2),
    "unique_countries": int(raw["Country"].nunique()),
    "sensitive_attributes": "Country (geographic origin) is the only quasi-demographic field; "
                             "no age, gender, income, or other personal demographic data is present.",
    "privacy_and_intended_use": "Transaction-level data with a numeric Customer ID (no name, "
                                 "address, or payment data). Used here strictly for an academic "
                                 "customer-segmentation exercise.",
    "direct_suitability_per_manual": "RESEARCH EXTENSION. No predefined segment label exists; "
                                      "a frozen label-construction procedure is required and is "
                                      "documented in label_definition.md.",
}
(OUT / "artifacts" / "dataset_card.json").write_text(json.dumps(dataset_card, indent=2))

# ---------------------------------------------------------------------------
# A3. Clean transaction-level data
# ---------------------------------------------------------------------------
raw["IsCancellation"] = raw["Invoice"].astype(str).str.startswith("C")
raw_valid_id = raw.dropna(subset=["Customer ID"]).copy()
raw_valid_id["Customer ID"] = raw_valid_id["Customer ID"].astype(int)

# Sales lines only (positive quantity, positive price, not a cancellation) are used for
# monetary/behavioral aggregation; cancellations are kept separately for the return-rate feature.
sales = raw_valid_id[(~raw_valid_id["IsCancellation"]) &
                      (raw_valid_id["Quantity"] > 0) &
                      (raw_valid_id["Price"] > 0)].copy()
sales["LineTotal"] = sales["Quantity"] * sales["Price"]
sales["IsWeekend"] = sales["InvoiceDate"].dt.dayofweek.isin([5, 6])

print("Sales line rows used:", sales.shape[0], "of", raw.shape[0])

# ---------------------------------------------------------------------------
# STAGE 1: Frozen RFM label construction (documented, not re-used as a predictor)
# ---------------------------------------------------------------------------
snapshot_date = raw_valid_id["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = sales.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("LineTotal", "sum"),
).reset_index()

# Keep customers with at least one qualifying purchase and positive monetary value
rfm = rfm[rfm["Monetary"] > 0].copy()

def score_quartile(series, ascending):
    ranks = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 4, labels=[1, 2, 3, 4]).astype(int)

rfm["R_score"] = score_quartile(rfm["Recency"], ascending=False)   # lower recency = higher score
rfm["F_score"] = score_quartile(rfm["Frequency"], ascending=True)  # higher frequency = higher score
rfm["M_score"] = score_quartile(rfm["Monetary"], ascending=True)   # higher monetary = higher score
rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def label_from_score(s):
    if s >= 10:
        return "Champions"
    elif s >= 7:
        return "Loyal Customers"
    elif s >= 4:
        return "At Risk"
    else:
        return "Hibernating or Lost"

rfm["customer_segment"] = rfm["RFM_total"].apply(label_from_score)

label_definition = """# Frozen Label-Construction Procedure (Stage 1)

Dataset: UCI Online Retail II. This dataset has NO predefined customer-segment label, so one
was constructed using a standard, documented RFM (Recency, Frequency, Monetary) scoring rule,
BEFORE any supervised model was trained. This rule was frozen and not modified after seeing
model results.

Snapshot date: max(InvoiceDate) + 1 day = {snapshot}

For each customer (sales lines only; cancellations excluded):
  Recency  = days since the customer's most recent purchase
  Frequency = number of distinct invoices
  Monetary  = total amount spent (Quantity x Price, summed)

Each of R, F, M is scored 1 (worst) to 4 (best) using quartiles across all customers.
RFM_total = R_score + F_score + M_score  (range 3 to 12)

Segment mapping (frozen business rule):
  RFM_total 10-12  -> Champions
  RFM_total 7-9    -> Loyal Customers
  RFM_total 4-6    -> At Risk
  RFM_total 3      -> Hibernating or Lost

## Label-Provenance and Circularity Audit (manual section 7.5)
Recency, Frequency, and Monetary (and their R/F/M scores) directly define this label by
construction. They are therefore EXCLUDED from the Stage 2 predictor set, because including
them would let the classifier trivially reconstruct the label it was derived from
(label circularity). Stage 2 instead uses an independent behavioral/demographic/psychographic
proxy feature set (country, average unit price, bulk-buy ratio, average basket size, product
diversity, return rate, tenure, weekend purchase ratio) that is correlated with, but not
deterministic of, the RFM label.

## Psychographic Measurement Provenance (manual section 7.6)
This dataset contains no self-reported psychographic survey data. The "average unit price"
and "bulk-buy ratio" features used in Stage 2 are BEHAVIORALLY INFERRED proxies for price
sensitivity and deal-seeking attitude, not directly measured psychographic constructs. They
should be interpreted with caution and are documented here as inferred, not ground truth.
""".format(snapshot=snapshot_date)

(OUT / "artifacts" / "label_definition.md").write_text(label_definition)
print(rfm["customer_segment"].value_counts())

# ---------------------------------------------------------------------------
# STAGE 2 feature engineering (non-circular predictors)
# ---------------------------------------------------------------------------
country_mode = sales.groupby("Customer ID")["Country"].agg(lambda x: x.mode().iloc[0])

basket = sales.groupby("Customer ID").agg(
    avg_unit_price=("Price", "mean"),
    avg_basket_size=("Quantity", "mean"),
    product_diversity=("StockCode", pd.Series.nunique),
    weekend_ratio=("IsWeekend", "mean"),
    first_purchase=("InvoiceDate", "min"),
    last_purchase=("InvoiceDate", "max"),
).reset_index()
basket["tenure_days"] = (basket["last_purchase"] - basket["first_purchase"]).dt.days

bulk = sales.copy()
bulk["is_bulk"] = bulk["Quantity"] >= 12
bulk_ratio = bulk.groupby("Customer ID")["is_bulk"].mean().rename("bulk_buy_ratio").reset_index()

all_invoices = raw_valid_id.groupby("Customer ID")["Invoice"].nunique().rename("all_invoice_count")
cancel_invoices = raw_valid_id[raw_valid_id["IsCancellation"]].groupby("Customer ID")["Invoice"].nunique().rename("cancel_invoice_count")
return_df = pd.concat([all_invoices, cancel_invoices], axis=1).fillna(0)
return_df["return_rate"] = return_df["cancel_invoice_count"] / return_df["all_invoice_count"].replace(0, np.nan)
return_df = return_df.reset_index()[["Customer ID", "return_rate"]]

customers = rfm[["Customer ID", "customer_segment", "Recency", "Frequency", "Monetary", "RFM_total"]].merge(
    country_mode.rename("country"), on="Customer ID", how="left"
).merge(
    basket[["Customer ID", "avg_unit_price", "avg_basket_size", "product_diversity",
            "weekend_ratio", "tenure_days"]], on="Customer ID", how="left"
).merge(
    bulk_ratio, on="Customer ID", how="left"
).merge(
    return_df, on="Customer ID", how="left"
)
customers["return_rate"] = customers["return_rate"].fillna(0.0)

# Collapse rare countries into "Other" so encoders behave sensibly
top_countries = customers["country"].value_counts().nlargest(8).index
customers["country"] = customers["country"].where(customers["country"].isin(top_countries), "Other")

customers.to_csv(OUT / "results" / "customer_level_dataset.csv", index=False)
print("Customer-level dataset shape:", customers.shape)

ID_COL = "Customer ID"
TARGET = "customer_segment"

DEMOGRAPHIC = ["country"]
PSYCHOGRAPHIC = ["avg_unit_price", "bulk_buy_ratio"]
BEHAVIORAL = ["avg_basket_size", "product_diversity", "weekend_ratio", "tenure_days", "return_rate"]
ALL_FEATURES = DEMOGRAPHIC + PSYCHOGRAPHIC + BEHAVIORAL

feature_manifest = {
    "demographic": DEMOGRAPHIC,
    "psychographic": PSYCHOGRAPHIC,
    "behavioral": BEHAVIORAL,
    "all_features": ALL_FEATURES,
    "excluded_circular_features": ["Recency", "Frequency", "Monetary", "RFM_total",
                                    "R_score", "F_score", "M_score"],
    "exclusion_reason": "These fields deterministically define customer_segment and were "
                         "excluded to prevent label circularity (see label_definition.md).",
}
(OUT / "artifacts" / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))
print(json.dumps(feature_manifest, indent=2))

df = customers.copy()
assert df[TARGET].nunique() >= 2
assert not df[ID_COL].duplicated().any()

# ---------------------------------------------------------------------------
# A4. Data-quality and class-distribution audit
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
    "dtype": df[ALL_FEATURES + [TARGET]].dtypes.astype(str),
    "missing_count": df[ALL_FEATURES + [TARGET]].isna().sum(),
    "missing_percent": 100 * df[ALL_FEATURES + [TARGET]].isna().mean(),
    "unique_count": df[ALL_FEATURES + [TARGET]].nunique(dropna=False),
})
summary.to_csv(OUT / "results" / "data_audit.csv")
print(summary)
print("Exact duplicate rows:", df.duplicated().sum())

class_counts = df[TARGET].value_counts().sort_index()
class_counts.to_csv(OUT / "results" / "class_distribution.csv")
ax = class_counts.plot(kind="bar", title="Customer Segment Distribution (RFM-derived)", color="#2c5f8a")
ax.set_xlabel("Segment"); ax.set_ylabel("Number of customers")
plt.tight_layout(); plt.savefig(OUT / "figures" / "class_distribution.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# A5. Create and save the locked split
# ---------------------------------------------------------------------------
usable = df.dropna(subset=[TARGET]).copy()
train_df, test_df = train_test_split(
    usable, test_size=0.20, random_state=SEED, stratify=usable[TARGET]
)
assert set(train_df[ID_COL]).isdisjoint(set(test_df[ID_COL]))
split_manifest = pd.concat([
    train_df[[ID_COL]].assign(split="train"),
    test_df[[ID_COL]].assign(split="test")
], ignore_index=True)
split_manifest.to_csv(OUT / "artifacts" / "split_manifest.csv", index=False)

X_train = train_df[ALL_FEATURES]; y_train = train_df[TARGET]
X_test = test_df[ALL_FEATURES]; y_test = test_df[TARGET]
print(X_train.shape, X_test.shape)

# ---------------------------------------------------------------------------
# A6. Type groups
# ---------------------------------------------------------------------------
numeric_cols = ["avg_unit_price", "bulk_buy_ratio", "avg_basket_size",
                 "product_diversity", "weekend_ratio", "tenure_days", "return_rate"]
nominal_cols = ["country"]
binary_cols = []
print("Numeric:", numeric_cols); print("Nominal:", nominal_cols)

# ---------------------------------------------------------------------------
# A7. Leakage-safe pipelines per model
# ---------------------------------------------------------------------------
class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    """Encode known categories as 1..K and reserve 0 for unseen categories."""
    def fit(self, X, y=None):
        self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.enc_.fit(X)
        return self
    def transform(self, X):
        return self.enc_.transform(X).astype(int) + 1

numeric_binary = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("bins", KBinsDiscretizer(n_bins=4, encode="onehot", strategy="quantile")),
])
category_ohe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
bernoulli_preprocessor = ColumnTransformer([
    ("num_bins", numeric_binary, numeric_cols),
    ("cat", category_ohe, nominal_cols + binary_cols),
], remainder="drop")

cat_num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("bins", KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")),
])
cat_cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("safe_ordinal", SafeOrdinalToNonNegative()),
])
categorical_nb_preprocessor = ColumnTransformer([
    ("num", cat_num, numeric_cols),
    ("cat", cat_cat, nominal_cols + binary_cols),
], remainder="drop")

core_pipelines = {
    "Dummy_most_frequent": Pipeline([("prep", bernoulli_preprocessor),
                                      ("model", DummyClassifier(strategy="most_frequent"))]),
    "BernoulliNB": Pipeline([("prep", bernoulli_preprocessor),
                              ("model", BernoulliNB(alpha=1.0, binarize=0.0))]),
    "CategoricalNB_mixed": Pipeline([("prep", categorical_nb_preprocessor),
                                      ("model", CategoricalNB(alpha=1.0))]),
}

gaussian_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])
core_pipelines["GaussianNB_numeric_only"] = Pipeline([
    ("prep", ColumnTransformer([("num", gaussian_preprocessor, numeric_cols)], remainder="drop")),
    ("model", GaussianNB(var_smoothing=1e-9)),
])

# Verify CategoricalNB non-negativity constraint on TRAINING data only
Xt_cat = categorical_nb_preprocessor.fit_transform(X_train)
assert np.nanmin(np.asarray(Xt_cat)) >= 0, "CategoricalNB received a negative category code."

# ---------------------------------------------------------------------------
# A10. Training-only cross-validation (identical folds, core models)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}

rows = []
for name, pipe in core_pipelines.items():
    start = time.perf_counter()
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                             return_train_score=False, n_jobs=1)
    elapsed = time.perf_counter() - start
    rows.append({
        "model": name,
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_sd": scores["test_accuracy"].std(ddof=1),
        "macro_f1_mean": scores["test_macro_f1"].mean(),
        "macro_f1_sd": scores["test_macro_f1"].std(ddof=1),
        "weighted_f1_mean": scores["test_weighted_f1"].mean(),
        "cv_time_seconds": elapsed,
    })
cv_results = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
cv_results.to_csv(OUT / "results" / "cv_results.csv", index=False)
print(cv_results)

ax = cv_results.set_index("model")[["accuracy_mean", "macro_f1_mean", "weighted_f1_mean"]].plot(
    kind="bar", figsize=(8, 5), title="Cross Validation Comparison (mean of 5 folds)")
plt.ylabel("Score"); plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.savefig(OUT / "figures" / "cv_comparison.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# A11. Locked-test evaluation (select BEFORE touching test set)
# ---------------------------------------------------------------------------
selected_name = cv_results.iloc[0]["model"]
selected_model = core_pipelines[selected_name]
print("Selected by mean CV macro F1:", selected_name)

start = time.perf_counter()
selected_model.fit(X_train, y_train)
train_seconds = time.perf_counter() - start

start = time.perf_counter()
y_pred = selected_model.predict(X_test)
inference_seconds = time.perf_counter() - start

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
pd.DataFrame(report).T.to_csv(OUT / "results" / "classification_report.csv")
print(classification_report(y_test, y_pred, zero_division=0))

test_summary = pd.DataFrame([{
    "model": selected_name,
    "accuracy": accuracy_score(y_test, y_pred),
    "macro_f1": f1_score(y_test, y_pred, average="macro"),
    "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    "train_seconds": train_seconds,
    "inference_seconds": inference_seconds,
    "test_n": len(y_test),
}])
test_summary.to_csv(OUT / "results" / "test_summary.csv", index=False)
print(test_summary)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=30, cmap="Blues")
plt.title(f"Confusion Matrix - {selected_name}")
plt.tight_layout(); plt.savefig(OUT / "figures" / "confusion_matrix.png", dpi=180); plt.close()

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
labels_sorted = sorted(y_test.unique())
pd.DataFrame(cm_norm, index=labels_sorted, columns=labels_sorted).to_csv(
    OUT / "results" / "confusion_matrix_normalized.csv")

# ---------------------------------------------------------------------------
# A12. Save predictions (the requested "result column") and artifacts
# ---------------------------------------------------------------------------
pred_df = test_df[[ID_COL, TARGET]].copy()
pred_df["predicted_segment"] = y_pred

if hasattr(selected_model, "predict_proba"):
    probs = selected_model.predict_proba(X_test)
    classes = selected_model.classes_
    pred_df["max_posterior"] = probs.max(axis=1)
    pred_df["confidence_category"] = pd.cut(
        pred_df["max_posterior"], bins=[-np.inf, .50, .75, np.inf],
        labels=["low_review", "moderate_review", "high"]
    )
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probs[:, i]
else:
    pred_df["max_posterior"] = np.nan
    pred_df["confidence_category"] = "decision_score_only"

pred_df.to_csv(OUT / "results" / "test_predictions.csv", index=False)

errors = pred_df[pred_df[TARGET] != pred_df["predicted_segment"]].copy()
errors = errors.merge(test_df[[ID_COL] + ALL_FEATURES], on=ID_COL, how="left")
errors.to_csv(OUT / "results" / "error_analysis.csv", index=False)
print("Test errors:", len(errors), "of", len(pred_df))

dump(selected_model, OUT / "models" / "selected_pipeline.joblib")

# ---------------------------------------------------------------------------
# Full-dataset scored output (every customer, with the result column)
# ---------------------------------------------------------------------------
full_X = df[ALL_FEATURES]
df_scored = df[[ID_COL, TARGET, "country"] + numeric_cols].copy()
df_scored["predicted_segment"] = selected_model.predict(full_X)
if hasattr(selected_model, "predict_proba"):
    full_probs = selected_model.predict_proba(full_X)
    df_scored["max_posterior"] = full_probs.max(axis=1)
df_scored["split"] = np.where(df_scored[ID_COL].isin(train_df[ID_COL]), "train", "test")
df_scored.to_csv(OUT / "results" / "full_customer_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# A13. New-customer prediction helper
# ---------------------------------------------------------------------------
REQUIRED_INPUTS = ALL_FEATURES
def predict_customer_segment(customer_profile: dict) -> dict:
    missing = [c for c in REQUIRED_INPUTS if c not in customer_profile]
    if missing:
        raise ValueError(f"Missing mandatory fields: {missing}")
    one = pd.DataFrame([customer_profile], columns=REQUIRED_INPUTS)
    pred = selected_model.predict(one)[0]
    result = {"predicted_segment": str(pred)}
    if hasattr(selected_model, "predict_proba"):
        p = selected_model.predict_proba(one)[0]
        distribution = {str(c): float(v) for c, v in zip(selected_model.classes_, p)}
        confidence = float(p.max())
        review = "normal_review" if confidence >= .75 else (
            "explicit_review" if confidence >= .50 else "manual_analysis")
        result.update({"posterior_distribution": distribution,
                        "max_posterior": confidence,
                        "review_recommendation": review})
    return result

example_profile = X_test.iloc[0].to_dict()
example_result = predict_customer_segment(example_profile)
(OUT / "artifacts" / "example_new_customer_prediction.json").write_text(json.dumps(example_result, indent=2))
print(example_result)

# ---------------------------------------------------------------------------
# A14. Core acceptance tests
# ---------------------------------------------------------------------------
assert TARGET in df.columns
assert df[TARGET].nunique() >= 2
assert ID_COL not in ALL_FEATURES
assert set(train_df[ID_COL]).isdisjoint(set(test_df[ID_COL]))
assert set(np.unique(y_pred)).issubset(set(y_train.unique()))
assert (OUT / "models" / "selected_pipeline.joblib").exists()
assert (OUT / "results" / "cv_results.csv").exists()
assert (OUT / "results" / "test_predictions.csv").exists()
required_core = {"Dummy_most_frequent", "BernoulliNB", "CategoricalNB_mixed", "GaussianNB_numeric_only"}
assert required_core.issubset(set(cv_results["model"]))
reloaded = __import__("joblib").load(OUT / "models" / "selected_pipeline.joblib")
assert np.array_equal(
    reloaded.predict(X_test.head(5)), selected_model.predict(X_test.head(5))
)
print("Core acceptance tests: PASSED")

readme = f"""# MDI3003 Lab 04 - Customer Segment Prediction (Online Retail II, Research Extension)

## What this run does
Dataset C (UCI Online Retail II) has no predefined segment label, so per the lab manual's
Dataset-use rule, a frozen two-stage procedure was used:

1. Stage 1: an RFM-based rule (documented in artifacts/label_definition.md) assigns every
   customer to one of 4 segments: Champions, Loyal Customers, At Risk, Hibernating or Lost.
2. Stage 2: four core classifiers (Dummy, GaussianNB, BernoulliNB, CategoricalNB) are trained
   with leakage-safe, training-only-fit pipelines to PREDICT that segment from a separate,
   non-circular feature set (country, avg unit price, bulk-buy ratio, avg basket size,
   product diversity, weekend ratio, tenure, return rate). Raw R/F/M values are excluded from
   the predictors to avoid label circularity.

## Selected model
{selected_name}, selected by mean 5-fold stratified cross-validation macro F1 on the
training split only, then evaluated once on the locked 20% test split.

## Key result files (results/)
- data_audit.csv               : missingness/uniqueness audit of the customer-level table
- class_distribution.csv       : segment counts
- cv_results.csv                : 5-fold CV comparison across all 4 core models
- classification_report.csv    : per-class precision/recall/F1 on the locked test set
- test_summary.csv              : accuracy, macro F1, weighted F1, timing on the locked test set
- confusion_matrix_normalized.csv
- test_predictions.csv          : RESULT COLUMN for the locked test customers
                                   (predicted_segment, max_posterior, confidence_category)
- full_customer_predictions.csv : RESULT COLUMN for every customer in the dataset (train+test)
- error_analysis.csv            : misclassified test customers with feature values

## Figures (figures/)
- class_distribution.png, cv_comparison.png, confusion_matrix.png

## Artifacts (artifacts/)
- label_definition.md   : frozen Stage-1 RFM label rule + circularity audit
- feature_manifest.json : demographic/psychographic/behavioral feature groups
- dataset_card.json     : dataset provenance, licence, privacy note
- split_manifest.csv    : which Customer ID went to train vs test
- versions.json          : package versions for reproducibility

## Models (models/)
- selected_pipeline.joblib : the fitted, deployable scikit-learn pipeline

## Limitations (read before using this for real decisions)
- The segment label is derived from the data itself (RFM rule), not an externally verified
  business segmentation, so it should be treated as an analytical construct, not ground truth.
- avg_unit_price and bulk_buy_ratio are behaviorally inferred psychographic proxies, not
  measured attitudes or survey data.
- Country is the only demographic-like field available in this transactional dataset.
- All data is UK-dominant (about 92% of rows), so segment definitions may not generalize well
  to other countries with very few customers.
"""
(OUT / "artifacts" / "README.md").write_text(readme)
print("\nDone. See lab04_outputs/ for all results.")
